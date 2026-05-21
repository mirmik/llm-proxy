"""Small stdout operation logger for proxy-level events."""

import time


def now():
    return time.perf_counter()


def elapsed_ms(start):
    return int((time.perf_counter() - start) * 1000)


def log_event(action, **fields):
    parts = [f"[op] {action}"]
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "yes" if value else "no"
        else:
            value = " ".join(str(value).split())
        parts.append(f"{key}={value}")
    print(" ".join(parts), flush=True)
