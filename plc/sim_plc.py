"""Simulated PLC: a Modbus TCP server with one relay coil and one counter per camera.

    coil[n-1]               alarm relay for camNN   (True = relay closed)
    holding register[n-1]   number of alarms raised for camNN

Run it (from repo root) and leave the window open:
    python -m plc.sim_plc
It prints a line whenever a relay or counter changes.

Built on pymodbus' SimDevice/SimData API (the older ModbusDeviceContext classes
are deprecated). State is read back over TCP with a normal Modbus client, which
is also how a real SCADA system would see this PLC.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time

from pymodbus.client import ModbusTcpClient
from pymodbus.server import ModbusTcpServer
from pymodbus.simulator import DataType, SimData, SimDevice

log = logging.getLogger("plc.sim")
CHANNELS = 8
DEVICE_ID = 1


class SimPLC:
    def __init__(self, host: str = "127.0.0.1", port: int = 5020, channels: int = CHANNELS):
        self.host, self.port, self.channels = host, port, channels
        n = channels
        self._device = SimDevice(id=DEVICE_ID, simdata=(
            [SimData(address=0, values=[False] * n, datatype=DataType.BITS)],      # coils: alarm relays
            [SimData(address=0, values=[False] * n, datatype=DataType.BITS)],      # discrete inputs (unused)
            [SimData(address=0, values=[0] * n, datatype=DataType.REGISTERS)],     # holding: alarm counters
            [SimData(address=0, values=[0] * n, datatype=DataType.REGISTERS)],     # input registers (unused)
        ))
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: ModbusTcpServer | None = None
        self._ready = threading.Event()
        # 0.0.0.0 means "listen on every interface"; to connect to ourselves use loopback
        self._reader = ModbusTcpClient("127.0.0.1" if host in ("", "0.0.0.0") else host, port=port, timeout=2)
        self._read_lock = threading.Lock()

    # -- state, as seen over the wire -------------------------------------
    def _read(self, fn):
        with self._read_lock:
            if not self._reader.connected:
                self._reader.connect()
            return fn()

    def coils(self) -> list[bool]:
        r = self._read(lambda: self._reader.read_coils(0, count=self.channels, device_id=DEVICE_ID))
        return [bool(b) for b in r.bits[:self.channels]]

    def registers(self) -> list[int]:
        r = self._read(lambda: self._reader.read_holding_registers(0, count=self.channels, device_id=DEVICE_ID))
        return list(r.registers)

    # -- lifecycle --------------------------------------------------------
    def start(self, timeout: float = 5.0) -> None:
        self._thread = threading.Thread(target=self._run, name="sim-plc", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("simulated PLC did not start")

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        # the server object must be created inside the running event loop
        self._server = ModbusTcpServer(self._device, address=(self.host, self.port))
        task = asyncio.create_task(self._server.serve_forever())
        await asyncio.sleep(0.3)  # give the socket time to bind before reporting ready
        self._ready.set()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        self._reader.close()
        if self._loop and self._server:
            asyncio.run_coroutine_threadsafe(self._server.shutdown(), self._loop).result(timeout=5)
        if self._thread:
            self._thread.join(timeout=5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 inside a container")
    ap.add_argument("--port", type=int, default=5020)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    plc = SimPLC(host=args.host, port=args.port)
    plc.start()
    print(f"Simulated PLC listening on {plc.host}:{plc.port}  (Ctrl+C to stop)", flush=True)
    prev_c, prev_r = plc.coils(), plc.registers()
    try:
        while True:
            time.sleep(0.1)
            c, r = plc.coils(), plc.registers()
            for i in range(plc.channels):
                if c[i] != prev_c[i] or r[i] != prev_r[i]:
                    print(f"{time.strftime('%H:%M:%S')}  cam{i + 1}: relay {'ON ' if c[i] else 'off'}  "
                          f"alarms so far: {r[i]}", flush=True)
            prev_c, prev_r = c, r
    except KeyboardInterrupt:
        pass
    finally:
        plc.stop()


if __name__ == "__main__":
    main()