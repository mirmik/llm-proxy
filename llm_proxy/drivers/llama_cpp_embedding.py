"""Dedicated llama.cpp runtime for dense text embeddings."""

from llm_proxy.drivers.base import RuntimeDriver


class LlamaCppEmbeddingDriver(RuntimeDriver):
    """Keep embedding processes distinct from chat-capable llama.cpp runtimes."""

    name = "llama_cpp_embedding"

    def kind(self):
        return "embedding"

    def runtime_meta(self, runtime_name, runtime, global_config):
        meta = super().runtime_meta(runtime_name, runtime, global_config)
        meta["context"] = int(runtime.get("context", 0)) or None
        return meta
