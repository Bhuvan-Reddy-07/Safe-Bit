"""
Thread-Safe In-Memory Sliding-Window Graph Store.
Maintains financial transaction edges and alternate device/IP telemetry in RAM.
Evicts transactions older than 30 minutes (1800s) to bound memory footprint.
Enables sub-millisecond neighborhood lookups without disk or DB I/O.
"""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from config.settings import settings


@dataclass
class TransactionRecord:
    tx_id: str
    src: str
    dst: str
    amount: float
    timestamp: float
    delta_t: float
    channel_idx: int
    device_fingerprint: str
    ip_subnet: str
    is_bot: bool
    is_mule: bool = False
    typology: str = "normal"


@dataclass
class AccountTelemetryNode:
    account_id: str
    first_seen: float
    last_inflow_time: float = 0.0
    last_outflow_time: float = 0.0
    recent_inflow_amount: float = 0.0
    recent_outflow_amount: float = 0.0
    devices: Set[str] = field(default_factory=set)
    subnets: Set[str] = field(default_factory=set)
    total_tx_count: int = 0
    is_merchant: bool = False


class InMemoryGraphStore:
    """
    High-performance in-memory temporal graph buffer for Safe-Bit streaming engine.
    Uses adjacency index lists and deque for bounded 30-minute sliding window.
    """

    def __init__(self, window_sec: float = settings.SLIDING_WINDOW_SEC):
        self.window_sec = window_sec
        self.lock = threading.RLock()

        # Edge storage
        self.edges: deque[TransactionRecord] = deque()

        # Temporal Adjacency indices: account_id -> deque of edge records
        self.out_adj: Dict[str, deque[TransactionRecord]] = defaultdict(deque)
        self.in_adj: Dict[str, deque[TransactionRecord]] = defaultdict(deque)

        # Account state and telemetry
        self.accounts: Dict[str, AccountTelemetryNode] = {}

        # Reverse lookup for telemetry collisions
        # device_hash -> set of account_ids using this device
        self.device_to_accounts: Dict[str, Set[str]] = defaultdict(set)
        # ip_subnet -> set of account_ids
        self.subnet_to_accounts: Dict[str, Set[str]] = defaultdict(set)

        self.last_eviction_time = 0.0

    def add_transaction(
        self,
        tx_id: str,
        src: str,
        dst: str,
        amount: float,
        timestamp: float,
        payment_channel: str,
        device_fingerprint: str,
        ip_subnet: str,
        is_bot: bool = False,
        is_mule: bool = False,
        typology: str = "normal",
    ) -> TransactionRecord:
        """
        Thread-safely appends a new transaction and updates account telemetry states.
        """
        with self.lock:
            # Channel one-hot index
            if payment_channel in settings.PAYMENT_CHANNELS:
                channel_idx = settings.PAYMENT_CHANNELS.index(payment_channel)
            else:
                channel_idx = 0

            # Compute delta_t: elapsed time since sender's previous incoming transaction
            src_node = self._get_or_create_account(src, timestamp)
            dst_node = self._get_or_create_account(dst, timestamp)

            if src_node.last_inflow_time > 0.0:
                delta_t = max(0.0, timestamp - src_node.last_inflow_time)
            else:
                delta_t = 0.0

            record = TransactionRecord(
                tx_id=tx_id,
                src=src,
                dst=dst,
                amount=amount,
                timestamp=timestamp,
                delta_t=delta_t,
                channel_idx=channel_idx,
                device_fingerprint=device_fingerprint,
                ip_subnet=ip_subnet,
                is_bot=is_bot,
                is_mule=is_mule,
                typology=typology,
            )

            # Update nodes
            src_node.last_outflow_time = timestamp
            src_node.recent_outflow_amount += amount
            src_node.devices.add(device_fingerprint)
            src_node.subnets.add(ip_subnet)
            src_node.total_tx_count += 1

            dst_node.last_inflow_time = timestamp
            dst_node.recent_inflow_amount += amount
            dst_node.total_tx_count += 1

            # Update indices
            self.edges.append(record)
            self.out_adj[src].append(record)
            self.in_adj[dst].append(record)

            self.device_to_accounts[device_fingerprint].add(src)
            self.subnet_to_accounts[ip_subnet].add(src)

            # Check for sliding-window eviction every 500 txs
            if len(self.edges) % 500 == 0:
                self.evict_expired(timestamp)

            return record

    def _get_or_create_account(self, account_id: str, current_time: float) -> AccountTelemetryNode:
        if account_id not in self.accounts:
            is_merchant = account_id.startswith("ACC_MERCHANT_")
            # If merchant, give baseline historical retention >48h
            first_seen = current_time - 200_000.0 if is_merchant else current_time
            self.accounts[account_id] = AccountTelemetryNode(
                account_id=account_id,
                first_seen=first_seen,
                is_merchant=is_merchant,
            )
        return self.accounts[account_id]

    def evict_expired(self, current_time: float):
        """
        Removes edges older than (current_time - window_sec).
        Maintains O(1) bounded RAM footprint.
        """
        cutoff = current_time - self.window_sec
        while self.edges and self.edges[0].timestamp < cutoff:
            old_tx = self.edges.popleft()
            # Clean from adjacencies in O(1)
            q_out = self.out_adj.get(old_tx.src)
            if q_out:
                if q_out and q_out[0].tx_id == old_tx.tx_id:
                    q_out.popleft()
                elif old_tx in q_out:
                    try:
                        q_out.remove(old_tx)
                    except ValueError:
                        pass
            q_in = self.in_adj.get(old_tx.dst)
            if q_in:
                if q_in and q_in[0].tx_id == old_tx.tx_id:
                    q_in.popleft()
                elif old_tx in q_in:
                    try:
                        q_in.remove(old_tx)
                    except ValueError:
                        pass

    def get_account_outflow_in_window(self, account_id: str, t_start: float, t_end: float) -> float:
        """Calculates total outflow from account within [t_start, t_end]."""
        with self.lock:
            txs = self.out_adj.get(account_id, [])
            return sum(tx.amount for tx in txs if t_start <= tx.timestamp <= t_end)

    def get_account_inflow_in_window(self, account_id: str, t_start: float, t_end: float) -> float:
        """Calculates total inflow to account within [t_start, t_end]."""
        with self.lock:
            txs = self.in_adj.get(account_id, [])
            return sum(tx.amount for tx in txs if t_start <= tx.timestamp <= t_end)

    def get_shared_telemetry_accounts(self, device: str, subnet: str) -> Set[str]:
        """Returns accounts sharing the given device fingerprint or IP subnet."""
        with self.lock:
            dev_accs = self.device_to_accounts.get(device, set())
            sub_accs = self.subnet_to_accounts.get(subnet, set())
            return set(dev_accs | sub_accs)

    def size(self) -> Tuple[int, int]:
        with self.lock:
            return len(self.accounts), len(self.edges)
