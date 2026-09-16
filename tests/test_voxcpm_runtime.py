import asyncio
import struct
import threading
import unittest
from types import SimpleNamespace

import aiohttp
from aiohttp.test_utils import TestClient, TestServer
from aiohttp import web

from llm_proxy.drivers import build_driver_registry
from llm_proxy.router import create_api_app
from llm_proxy.runners.voxcpm_server import build_app


PCM = struct.pack('<fff', 0., .25, -.25)


class FakeEngine:
    sample_rate = 48000

    def __init__(self):
        self.release = threading.Event()
        self.closed = threading.Event()
        self.fail_before = False
        self.fail_after = False

    def generate(self, text):
        try:
            if self.fail_before:
                raise RuntimeError('GPU failure')
            yield PCM
            self.release.wait(3)
            if self.fail_after:
                raise RuntimeError('GPU failed mid-stream')
            yield PCM
        finally:
            self.closed.set()


class FakeManager:
    def __init__(self, port):
        self.port = port
        self.running = True
        self.ensure_calls = 0

    def runtimes_by_kind(self, kind):
        return ['tts-main', 'voxcpm2']

    def is_running(self, name):
        return self.running and name == 'voxcpm2'

    def on_demand(self, name):
        return False

    def get_port(self, name):
        return self.port

    async def ensure(self, name):
        self.ensure_calls += 1


class VoxCPMTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = FakeEngine()
        self.backend = TestServer(build_app(SimpleNamespace(max_text_length=500), self.engine),
                                  handler_cancellation=True)
        await self.backend.start_server()
        self.rm = FakeManager(self.backend.port)
        self.proxy = TestClient(TestServer(create_api_app(self.rm, None)))
        self.direct = TestClient(self.backend)
        await self.direct.start_server()
        await self.proxy.start_server()

    async def asyncTearDown(self):
        self.engine.release.set()
        await self.proxy.close()
        await self.direct.close()

    async def test_stream_reaches_client_before_generation_finishes(self):
        resp = await self.proxy.post('/v1/audio/speech', json={
            'voice': 'voxcpm2', 'text': 'Привет.', 'stream': True})
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.headers['X-Audio-Sample-Rate'], '48000')
        self.assertEqual(resp.headers['X-Audio-Format'], 'f32le')
        self.assertEqual(await asyncio.wait_for(resp.content.readexactly(len(PCM)), 1), PCM)
        self.assertFalse(self.engine.closed.is_set())
        health = await asyncio.wait_for(self.proxy.get('/tts/health?voice=voxcpm2'), 1)
        self.assertTrue((await health.json())['busy'])
        other = await self.proxy.post('/v1/audio/speech', json={'voice': 'voxcpm2', 'text': 'Ещё.'})
        self.assertEqual(other.status, 503)
        self.engine.release.set()
        self.assertEqual(await resp.read(), PCM)

    async def test_full_response_has_same_pcm_contract(self):
        self.engine.release.set()
        resp = await self.proxy.post('/v1/audio/speech', json={'voice': 'voxcpm2', 'text': 'Привет.'})
        self.assertEqual(resp.status, 200)
        self.assertEqual(await resp.read(), PCM * 2)

    async def test_stopped_returns_503_and_never_loads_on_request(self):
        self.rm.running = False
        for path, method in [('/tts/health?voice=voxcpm2', self.proxy.get),
                             ('/v1/audio/speech', self.proxy.post)]:
            kwargs = {'json': {'voice': 'voxcpm2', 'text': 'Привет.'}} if method == self.proxy.post else {}
            resp = await method(path, **kwargs)
            self.assertEqual(resp.status, 503)
        self.assertEqual(self.rm.ensure_calls, 0)

    async def test_unknown_voice_does_not_select_another_model(self):
        for voice in ['missing', '']:
            resp = await self.proxy.post('/v1/audio/speech', json={'voice': voice, 'text': 'Привет.'})
            self.assertEqual(resp.status, 404 if voice else 400)

    async def test_invalid_requests_fail_without_starting_gpu(self):
        for body in [[], {'text': ''}, {'text': 12}, {'text': 'x' * 501},
                     {'text': 'Hi', 'stream': 'true'}, {'text': 'Hi', 'speed': 2}]:
            resp = await self.direct.post('/tts/generate', json=body)
            self.assertEqual(resp.status, 400, body)
        self.assertFalse(self.engine.closed.is_set())

    async def test_engine_failure_before_headers_is_502(self):
        self.engine.fail_before = True
        resp = await self.proxy.post('/v1/audio/speech', json={'voice': 'voxcpm2', 'text': 'Hi', 'stream': True})
        self.assertEqual(resp.status, 502)

    async def test_partial_failure_is_not_reported_as_complete_audio(self):
        self.engine.fail_after = True
        resp = await self.proxy.post('/v1/audio/speech', json={'voice': 'voxcpm2', 'text': 'Hi', 'stream': True})
        self.assertEqual(await resp.content.readexactly(len(PCM)), PCM)
        self.engine.release.set()
        with self.assertRaises(aiohttp.ClientPayloadError):
            await resp.read()

    async def test_disconnect_releases_engine(self):
        resp = await self.direct.post('/tts/generate', json={'text': 'Hi', 'stream': True})
        await resp.content.readexactly(len(PCM))
        resp.close()
        self.engine.release.set()
        self.assertTrue(await asyncio.to_thread(self.engine.closed.wait, 2))
        for _ in range(50):
            health = await self.direct.get('/health')
            if not (await health.json())['busy']:
                break
            await asyncio.sleep(.01)
        else:
            self.fail('GPU stayed busy after disconnect')

    async def test_backend_health_failure_is_propagated(self):
        async def failed(request):
            return web.json_response({'status': 'failed'}, status=503)
        app = web.Application()
        app.router.add_get('/health', failed)
        async with TestServer(app) as server:
            self.rm.port = server.port
            resp = await self.proxy.get('/tts/health?voice=voxcpm2')
            self.assertEqual(resp.status, 503)
            self.assertFalse((await resp.json())['loaded'])


class DriverTests(unittest.TestCase):
    def test_voxcpm_command_uses_its_own_environment(self):
        driver = build_driver_registry()['voxcpm']
        cfg = {'port': 5052, 'venv': '/tmp/tts-env',
               'command': '{python} {runners_dir}/voxcpm_server.py --port {port}'}
        driver.validate_runtime(cfg)
        self.assertEqual(driver.kind(), 'tts')
        argv = driver.build_command('voxcpm2', cfg, {})
        self.assertEqual(argv[0], '/tmp/tts-env/bin/python')
        self.assertEqual(argv[-1], '5052')
