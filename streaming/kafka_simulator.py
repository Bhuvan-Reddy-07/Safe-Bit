"""
High-Throughput Kafka Stream Simulator.
Simulates high-velocity financial streams with realistic traffic patterns,
bursts of money mule rings, smurfing fan-outs, and peeling chains.
"""

import asyncio
import queue
import random
import threading
import time
from typing import AsyncGenerator, Generator, List, Optional

from data.synthetic_alt_generator import (
    SyntheticTelemetryGenerator,
    TransactionEvent,
    synthetic_generator,
)


class MockTransactionStreamer:
    """
    Simulates Apache Kafka topic partitions for streaming transaction ingestion.
    Supports both synchronous generator and asynchronous coroutine streaming.
    """

    def __init__(
        self,
        tps: float = 100.0,
        mule_ratio: float = 0.05,
        generator: Optional[SyntheticTelemetryGenerator] = None,
    ):
        self.tps = tps
        self.mule_ratio = mule_ratio
        self.gen = generator or synthetic_generator
        self.tx_queue: queue.Queue = queue.Queue(maxsize=10000)
        self.is_running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Starts background producer thread pushing into simulated Kafka topic."""
        self.is_running = True
        self._thread = threading.Thread(target=self._producer_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _producer_loop(self):
        current_time = time.time()
        while self.is_running:
            # Batch of transactions
            batch = self.gen.generate_batch_dataset(num_transactions=50, mule_ratio=self.mule_ratio)
            for tx in batch:
                if not self.is_running:
                    break
                try:
                    self.tx_queue.put(tx, timeout=0.1)
                except queue.Full:
                    pass
                time.sleep(1.0 / max(1.0, self.tps))

    def poll_sync(self, timeout: float = 0.1) -> Optional[TransactionEvent]:
        try:
            return self.tx_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    async def stream_async(self, limit: Optional[int] = None) -> AsyncGenerator[TransactionEvent, None]:
        """Async generator yielding transactions for FastAPI websocket / streaming endpoints."""
        count = 0
        base_t = time.time()
        while limit is None or count < limit:
            # Generate single event on the fly
            roll = random.random()
            if roll < self.mule_ratio:
                ring = self.gen.generate_mule_ring(base_t, ring_size=random.randint(4, 6))
                for tx in ring:
                    yield tx
                    count += 1
                    await asyncio.sleep(1.0 / self.tps)
            else:
                tx = self.gen.generate_legit_p2p(base_t)
                yield tx
                count += 1
                await asyncio.sleep(1.0 / self.tps)
            base_t = time.time()
