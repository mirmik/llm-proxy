"""Shared GPU/resource conflict coordinator."""

from llm_proxy.oplog import log_event


class ResourceCoordinator:
    """Stops conflicting managed items before a new one starts."""

    def __init__(self):
        self._handlers = {}

    def register(self, kind, list_running_on_gpu, stop_item):
        self._handlers[kind] = {
            "list": list_running_on_gpu,
            "stop": stop_item,
        }

    async def prepare(self, kind, name, gpu):
        if not gpu:
            return
        for other_kind, handler in self._handlers.items():
            for other_name in handler["list"](gpu):
                if other_kind == kind and other_name == name:
                    continue
                log_event(
                    "resource conflict stop",
                    requested=f"{kind}:{name}",
                    stopped=f"{other_kind}:{other_name}",
                    gpu=gpu,
                )
                await handler["stop"](other_name)
