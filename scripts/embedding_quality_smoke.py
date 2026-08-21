#!/usr/bin/env python3
"""Small multilingual quality and response-contract smoke test."""

import argparse
import json
import math
import os
import time
import urllib.request


PHRASES = [
    "зависание после перезагрузки Python-модуля",
    "editor freezes after Python module hot reload",
    "обновить TLS-сертификат",
    "карточка заблокирована зависимостью",
    "task is blocked by another task",
    "ускорить компиляцию шейдеров",
]


def cosine(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8090/v1/embeddings")
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument("--token", default=os.environ.get("LLM_PROXY_TOKEN"))
    args = parser.parse_args()

    body = json.dumps({"model": args.model, "input": PHRASES}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
    request = urllib.request.Request(args.url, data=body, headers=headers, method="POST")

    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
        status = response.status
    duration_ms = round((time.perf_counter() - started) * 1000, 1)

    data = payload.get("data") or []
    indices = [item.get("index") for item in data]
    vectors = [item.get("embedding") for item in data]
    if status != 200 or len(vectors) != len(PHRASES) or not all(isinstance(vector, list) for vector in vectors):
        raise SystemExit(f"unexpected response: status={status}, vectors={len(vectors)}")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or indices != list(range(len(PHRASES))):
        raise SystemExit(f"invalid dimensions/order: dimensions={dimensions}, indices={indices}")
    if not all(math.isfinite(value) for vector in vectors for value in vector):
        raise SystemExit("vectors contain non-finite values")

    norms = [math.sqrt(sum(value * value for value in vector)) for vector in vectors]
    if not all(abs(norm - 1.0) < 1e-3 for norm in norms):
        raise SystemExit(f"vectors are not L2-normalized: {min(norms):.6f}..{max(norms):.6f}")
    hot_reload_related = cosine(vectors[0], vectors[1])
    hot_reload_unrelated = cosine(vectors[0], vectors[2])
    blocked_related = cosine(vectors[3], vectors[4])
    blocked_unrelated = cosine(vectors[3], vectors[5])
    if hot_reload_related <= hot_reload_unrelated or blocked_related <= blocked_unrelated:
        raise SystemExit("multilingual relative ranking check failed")

    print(f"status={status} duration_ms={duration_ms} vectors={len(vectors)} dim={dimensions.pop()}")
    print(f"norm_range={min(norms):.6f}..{max(norms):.6f} finite=true")
    print(f"hot_reload: related={hot_reload_related:.4f} unrelated={hot_reload_unrelated:.4f}")
    print(f"blocked_task: related={blocked_related:.4f} unrelated={blocked_unrelated:.4f}")


if __name__ == "__main__":
    main()
