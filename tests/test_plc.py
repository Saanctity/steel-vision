import socket
import time

import pytest

from plc.modbus_alarm import ModbusAlarm, channel_for
from plc.sim_plc import SimPLC


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_channel_for():
    assert channel_for("cam1") == 0 and channel_for("cam12") == 11
    for bad in ("camera", "cam0", ""):
        with pytest.raises(ValueError):
            channel_for(bad)


def test_relay_closes_counts_and_reopens():
    port = _free_port()
    plc = SimPLC(port=port)
    plc.start()
    alarm = ModbusAlarm(port=port, hold_s=0.4)
    try:
        alarm.trigger("cam2")
        assert _wait(lambda: plc.coils()[1] is True)       # relay closed on channel 1
        assert plc.coils()[0] is False                      # other cameras untouched
        assert _wait(lambda: plc.registers()[1] == 1)       # alarm counter incremented
        assert _wait(lambda: plc.coils()[1] is False)       # pulse ended after hold_s
        alarm.trigger("cam2")
        assert _wait(lambda: plc.registers()[1] == 2)
    finally:
        alarm.close()
        plc.stop()


def test_trigger_never_blocks_or_raises_when_plc_is_down_and_recovers_later():
    port = _free_port()
    alarm = ModbusAlarm(port=port, hold_s=0.2, timeout=0.3)
    try:
        t0 = time.monotonic()
        alarm.trigger("cam1")                               # nothing is listening
        assert time.monotonic() - t0 < 0.1                  # returned immediately
        assert _wait(lambda: alarm.errors >= 1)             # failure was counted, not raised
        plc = SimPLC(port=port)
        plc.start()
        try:
            alarm.trigger("cam1")                           # PLC is back: reconnects by itself
            assert _wait(lambda: plc.registers()[0] >= 1, timeout=8)
        finally:
            plc.stop()
    finally:
        alarm.close()

    

def test_stale_alarms_are_dropped_not_sent_late():
    port = _free_port()
    plc = SimPLC(port=port)
    plc.start()
    alarm = ModbusAlarm(port=port, hold_s=0.2, max_age_s=1.0)
    try:
        alarm._q.put(("cam1", time.monotonic() - 10))      # an alarm that waited 10 s in a backlog
        assert _wait(lambda: alarm.dropped == 1)
        assert plc.registers()[0] == 0 and plc.coils()[0] is False   # it must NOT reach the PLC
        alarm.trigger("cam1")                              # a fresh alarm still works
        assert _wait(lambda: plc.registers()[0] == 1)
        assert alarm.dropped == 1
    finally:
        alarm.close()
        plc.stop()