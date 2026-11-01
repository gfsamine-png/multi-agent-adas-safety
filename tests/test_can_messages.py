import pytest

from adas.ecu.can_messages import (
    FCW_ID, LDW_ID, CanFrame, decode_fcw, decode_ldw, encode_fcw, encode_ldw,
)


def test_fcw_round_trip():
    frame = encode_fcw(5, "high", 21, 2.4)
    assert frame.arbitration_id == FCW_ID
    assert len(frame.data) == 8
    msg = decode_fcw(frame)
    assert (msg.counter, msg.level, msg.track_id) == (5, "high", 21)
    assert msg.ttc_s == pytest.approx(2.4)


def test_fcw_without_ttc_decodes_to_none():
    assert decode_fcw(encode_fcw(0, "unknown", 3, None)).ttc_s is None


def test_fcw_large_ttc_is_clamped_not_wrapped():
    assert decode_fcw(encode_fcw(0, "low", 1, 1e6)).ttc_s == pytest.approx(6553.4)


def test_counter_wraps_at_256():
    assert decode_fcw(encode_fcw(256 + 3, "low", 1, 5.0)).counter == 3


def test_corrupted_byte_is_rejected():
    frame = encode_fcw(1, "critical", 7, 1.0)
    bad = bytearray(frame.data)
    bad[1] ^= 0x01
    with pytest.raises(ValueError):
        decode_fcw(CanFrame(frame.arbitration_id, bytes(bad)))


def test_wrong_id_is_rejected():
    frame = encode_fcw(1, "low", 1, 5.0)
    with pytest.raises(ValueError):
        decode_ldw(frame)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        encode_fcw(0, "banana", 1, 2.0)
    with pytest.raises(ValueError):
        encode_fcw(0, "low", 1, -1.0)
    with pytest.raises(ValueError):
        encode_fcw(0, "low", 70000, 2.0)


def test_ldw_round_trip_left_and_right():
    left = decode_ldw(encode_ldw(9, "left", -0.4))
    assert (left.side, left.counter) == ("left", 9)
    assert left.offset == pytest.approx(-0.4)
    right = decode_ldw(encode_ldw(10, "right", 0.352))
    assert right.side == "right"
    assert right.offset == pytest.approx(0.352)
    assert encode_ldw(0, "left", -0.4).arbitration_id == LDW_ID


def test_ldw_rejects_bad_input():
    with pytest.raises(ValueError):
        encode_ldw(0, "up", 0.3)
    with pytest.raises(ValueError):
        encode_ldw(0, "left", float("nan"))