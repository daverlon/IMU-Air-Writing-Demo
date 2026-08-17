"""BLE client for the Symly gesture band (same framing as Gesture Recording Studio)."""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass
from typing import Callable, Optional

from bleak import BleakClient, BleakScanner

UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
SAMPLE_SIZE = 40


@dataclass
class IMUSample:
    timestamp_ms: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    roll: float
    pitch: float
    yaw: float


class GestureBandClient:
    def __init__(self, on_sample: Callable[[IMUSample], None]):
        self.client: Optional[BleakClient] = None
        self.on_sample = on_sample
        self.buffer = bytearray()

    async def find_device(self) -> Optional[str]:
        print("Scanning for gesture band (SY marker / BAND)...")
        found: dict[str, str] = {}

        def callback(device, advertisement_data):
            mfg = advertisement_data.manufacturer_data or {}
            for company_id, data in mfg.items():
                if company_id == 0xFFFF and len(data) >= 2 and data[0] == 0x53 and data[1] == 0x59:
                    found[device.address] = device.name or "BAND"
                    return
            if device.name == "BAND":
                found[device.address] = device.name

        scanner = BleakScanner(callback)
        await scanner.start()
        await asyncio.sleep(5.0)
        await scanner.stop()

        if not found:
            print("No gesture band found")
            return None
        address, name = next(iter(found.items()))
        print(f"Found {name} @ {address}")
        return address

    def _parse_samples(self, _sender, data: bytearray):
        self.buffer.extend(data)
        while len(self.buffer) >= SAMPLE_SIZE:
            chunk = self.buffer[:SAMPLE_SIZE]
            del self.buffer[:SAMPLE_SIZE]
            try:
                values = struct.unpack("<I9f", chunk)
            except struct.error:
                continue
            self.on_sample(
                IMUSample(
                    timestamp_ms=values[0],
                    gx=values[1],
                    gy=values[2],
                    gz=values[3],
                    ax=values[4],
                    ay=values[5],
                    az=values[6],
                    roll=values[7],
                    pitch=values[8],
                    yaw=values[9],
                )
            )

    async def connect(self, address: Optional[str] = None):
        if address is None:
            address = await self.find_device()
            if address is None:
                raise RuntimeError("Could not find gesture band")
        print(f"Connecting to {address}...")
        self.client = BleakClient(address)
        await self.client.connect()
        print("Connected")
        await self.client.start_notify(UART_TX_CHAR_UUID, self._parse_samples)

    async def disconnect(self):
        if self.client and self.client.is_connected:
            try:
                await self.client.stop_notify(UART_TX_CHAR_UUID)
            except Exception:
                pass
            await self.client.disconnect()

    async def run_forever(self):
        try:
            while True:
                await asyncio.sleep(1)
                if self.client and not self.client.is_connected:
                    print("Disconnected")
                    break
        finally:
            await self.disconnect()
