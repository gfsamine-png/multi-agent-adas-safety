"""Custom CAN message format for the ADAS prototype (alerts only).

IDs and layouts are our own, documented in the README. They are not a real
vehicle DBC. All frames are 8 bytes; byte 0 is a rolling counter and byte 7
is the XOR of bytes 0-6.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass

FCW_ID = 0x300
LDW_ID = 0x301
TTC_NOT_AVAILABLE = 0xFFFF

LEVEL_CODES = {"unknown": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}
LEVEL_NAMES = {v: k for k, v in LEVEL_CODES.items()}
SIDE_CODES = {"left": 1, "right": 2}
SIDE_NAMES = {v: k for k, v in SIDE_CODES.items()}


@dataclass(frozen=True)
class CanFrame:
    arbitration_id: int
    data: bytes  # always 8 bytes


@dataclass(frozen=True)
class FcwMessage:
    counter: int
    level: str
    track_id: int
    ttc_s: float | None


@dataclass(frozen=True)
class LdwMessage:
    counter: int
    side: str
    offset: float


def _checksum(body: bytes) -> int:
    x = 0
    for v in body:
        x ^= v
    return x


def _seal(arbitration_id: int, body: bytes) -> CanFrame:
    assert len(body) == 7
    return CanFrame(arbitration_id, body + bytes([_checksum(body)]))


def _open(frame: CanFrame, expected_id: int) -> bytes:
    if frame.arbitration_id != expected_id:
        raise ValueError(f"unexpected id 0x{frame.arbitration_id:X}")
    if len(frame.data) != 8:
        raise ValueError("CAN data must be 8 bytes")
    if _checksum(frame.data[:7]) != frame.data[7]:
        raise ValueError("checksum mismatch")
    return frame.data[:7]


def encode_fcw(counter: int, level: str, track_id: int, ttc_s: float | None) -> CanFrame:
    if level not in LEVEL_CODES:
        raise ValueError(f"unknown risk level: {level}")
    if not 0 <= track_id <= 0xFFFF:
        raise ValueError("track_id must fit in 16 bits")
    if ttc_s is None:
        ttc_raw = TTC_NOT_AVAILABLE
    else:
        if not math.isfinite(ttc_s) or ttc_s < 0:
            raise ValueError("ttc_s must be a finite value >= 0")
        ttc_raw = min(int(round(ttc_s * 10)), TTC_NOT_AVAILABLE - 1)
    body = struct.pack("<BBHHB", counter & 0xFF, LEVEL_CODES[level], track_id, ttc_raw, 0)
    return _seal(FCW_ID, body)


def decode_fcw(frame: CanFrame) -> FcwMessage:
    body = _open(frame, FCW_ID)
    counter, level, track_id, ttc_raw, _ = struct.unpack("<BBHHB", body)
    if level not in LEVEL_NAMES:
        raise ValueError(f"unknown level code: {level}")
    ttc = None if ttc_raw == TTC_NOT_AVAILABLE else ttc_raw / 10.0
    return FcwMessage(counter, LEVEL_NAMES[level], track_id, ttc)


def encode_ldw(counter: int, side: str, offset: float) -> CanFrame:
    if side not in SIDE_CODES:
        raise ValueError(f"unknown side: {side}")
    if not math.isfinite(offset):
        raise ValueError("offset must be finite")
    raw = max(-32767, min(32767, int(round(offset * 1000))))
    body = struct.pack("<BBh3x", counter & 0xFF, SIDE_CODES[side], raw)
    return _seal(LDW_ID, body)


def decode_ldw(frame: CanFrame) -> LdwMessage:
    body = _open(frame, LDW_ID)
    counter, side, raw = struct.unpack("<BBh3x", body)
    if side not in SIDE_NAMES:
        raise ValueError(f"unknown side code: {side}")
    return LdwMessage(counter, SIDE_NAMES[side], raw / 1000.0)