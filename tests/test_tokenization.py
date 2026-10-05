import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from llm_proxy.router import create_api_app
from llm_proxy.route_manager import RouteManager
from tests.test_embedding_runtime import FakeRuntimeManager, FakeSlotManager, DenyAuth


class TokenizationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.clients = []

    async def asyncTearDown(self):
        for client in reversed(self.clients):
            await client.close()

    async def start(self, app):
        client = TestClient(TestServer(app))
        await client.start_server()
        self.clients.append(client)
        return client

    async def backend(self, status=200):
        self.received = []
        async def tokenize(request):
            self.received.append((await request.json(), request.headers.get('Authorization')))
            return web.json_response({'tokens': [1, 2, 3]}, status=status)
        app = web.Application()
        app.router.add_post('/tokenize', tokenize)
        return await self.start(app)

    async def test_local_loaded_runtime_tokenizes_without_slot_or_start(self):
        backend = await self.backend()
        rm = FakeRuntimeManager(chat_names=['qwen'], ports={'qwen': backend.server.port})
        client = await self.start(create_api_app(rm, FakeSlotManager()))
        response = await client.post('/tokenize', json={'model': 'qwen', 'content': 'hello'})
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.json(), {'tokens': [1, 2, 3]})
        self.assertEqual(rm.ensure_calls, [])
        self.assertEqual(self.received, [({'model': 'qwen', 'content': 'hello'}, None)])

    async def test_route_rewrites_model_and_forwards_backend_auth(self):
        backend = await self.backend()
        routes = RouteManager({'hosts': {'worker': {'token': 'upstream-key'}}, 'routes': {
            'route': {'kind': 'llm', 'backends': [{
                'url': str(backend.make_url('/')).rstrip('/'), 'model': 'upstream',
                'auth': 'worker',
            }]},
        }})
        client = await self.start(create_api_app(FakeRuntimeManager(), FakeSlotManager(), route_manager=routes))
        response = await client.post('/tokenize', json={'model': 'route', 'content': 'hello'})
        self.assertEqual(response.status, 200)
        self.assertEqual(self.received, [({'model': 'upstream', 'content': 'hello'}, 'Bearer upstream-key')])

    async def test_validation_auth_and_unloaded_model(self):
        rm = FakeRuntimeManager(chat_names=['qwen'], on_demand=['qwen'])
        client = await self.start(create_api_app(rm, FakeSlotManager()))
        for body in [[], {}, {'model': '', 'content': 'x'}, {'model': 'qwen', 'content': 10}]:
            response = await client.post('/tokenize', json=body)
            self.assertEqual(response.status, 400)
        response = await client.post('/tokenize', data='{')
        self.assertEqual(response.status, 400)
        response = await client.post('/tokenize', json={'model': 'qwen', 'content': 'hello'})
        self.assertEqual(response.status, 503)
        self.assertEqual(rm.ensure_calls, [])
        response = await client.post('/tokenize', json={'model': 'missing', 'content': 'x'})
        self.assertEqual(response.status, 404)
        secured = await self.start(create_api_app(rm, FakeSlotManager(), DenyAuth()))
        response = await secured.post('/tokenize', json={'model': 'qwen', 'content': 'hello'})
        self.assertEqual(response.status, 401)

    async def test_unsupported_tokenizer_does_not_mark_chat_unhealthy(self):
        backend = await self.backend(status=404)
        routes = RouteManager({'routes': {'route': {'kind': 'llm', 'backends': [{
            'url': str(backend.make_url('/')).rstrip('/'), 'model': 'upstream',
        }]}}})
        route = routes.get_route('route')
        async def forbidden(*args):
            self.fail('Tokenization must not mark chat unhealthy')
        route.mark_unhealthy = forbidden
        client = await self.start(create_api_app(FakeRuntimeManager(), FakeSlotManager(), route_manager=routes))
        response = await client.post('/tokenize', json={'model': 'route', 'content': 'hello'})
        self.assertEqual(response.status, 404)
