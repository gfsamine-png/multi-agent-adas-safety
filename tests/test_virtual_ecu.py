from adas.ecu.can_messages import CanFrame, encode_fcw, encode_ldw
from adas.ecu.virtual_ecu import DashboardEcu


def test_valid_collision_frame_is_shown():
    text = DashboardEcu().receive(encode_fcw(0, "high", 21, 2.4))
    assert "COLLISION WARNING HIGH" in text
    assert "#21" in text


def test_valid_lane_frame_is_shown():
    text = DashboardEcu().receive(encode_ldw(0, "left", -0.4))
    assert "LANE DEPARTURE LEFT" in text


def test_lost_messages_are_detected():
    ecu = DashboardEcu()
    ecu.receive(encode_fcw(3, "low", 1, 5.0))
    assert "2 message(s) lost" in ecu.receive(encode_fcw(6, "low", 1, 5.0))


def test_counter_wrap_is_not_a_loss():
    ecu = DashboardEcu()
    ecu.receive(encode_fcw(255, "low", 1, 5.0))
    assert "lost" not in ecu.receive(encode_fcw(0, "low", 1, 5.0))


def test_repeated_counter_is_flagged():
    ecu = DashboardEcu()
    ecu.receive(encode_fcw(4, "low", 1, 5.0))
    assert "frozen" in ecu.receive(encode_fcw(4, "low", 1, 5.0))


def test_corrupted_frame_is_rejected_and_counted():
    ecu = DashboardEcu()
    frame = encode_fcw(1, "critical", 7, 1.0)
    bad = bytearray(frame.data)
    bad[2] ^= 0xFF
    assert ecu.receive(CanFrame(frame.arbitration_id, bytes(bad))).startswith("REJECTED")
    assert ecu.rejected == 1


def test_unknown_id_is_rejected():
    assert DashboardEcu().receive(CanFrame(0x7FF, bytes(8))).startswith("REJECTED")