from __future__ import annotations

def transfer_seconds(payload_bytes: int, bandwidth_mbps: float) -> float:
    if bandwidth_mbps <= 0:
        raise ValueError("bandwidth_mbps must be positive")
    return payload_bytes * 8.0 / (bandwidth_mbps * 1_000_000.0)

def report(payload_bytes: int, bandwidths=(5, 10, 25, 50, 100)):
    return {
        f"{float(b):g}_mbps_seconds": transfer_seconds(payload_bytes, float(b))
        for b in bandwidths
    }
