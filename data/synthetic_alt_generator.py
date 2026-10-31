"""
Synthetic Alternate Telemetry & Financial Transaction Stream Generator.
Simulates high-velocity banking and crypto transactions enriched with alternate telemetry:
  - Financial Edges (E_fin): amount (log1p), delta_t, one-hot rails
  - Telemetry Edges (E_alt): device_fingerprint_hash, ip_subnet_asn, cadence_bot_flag
  - Typologies: Mule rings, fan-out layering (smurfing), peeling chains,
    and synthetic legitimate high-volume merchants.
"""

import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from config.settings import settings


@dataclass
class TransactionEvent:
    tx_id: str
    src_account: str
    dst_account: str
    amount: float
    timestamp: float
    payment_channel: str
    device_fingerprint: str
    ip_subnet: str
    cadence_ms: float
    cadence_bot_flag: bool
    is_mule: bool
    typology: str  # 'ring', 'smurfing', 'peeling', 'merchant', 'legit_p2p'
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "tx_id": self.tx_id,
            "src_account": self.src_account,
            "dst_account": self.dst_account,
            "amount": round(self.amount, 2),
            "timestamp": round(self.timestamp, 4),
            "payment_channel": self.payment_channel,
            "device_fingerprint": self.device_fingerprint,
            "ip_subnet": self.ip_subnet,
            "cadence_ms": round(self.cadence_ms, 2),
            "cadence_bot_flag": self.cadence_bot_flag,
            "is_mule": self.is_mule,
            "typology": self.typology,
        }


class SyntheticTelemetryGenerator:
    """
    Generates realistic alternate telemetry and financial transaction streams.
    Maintains simulated account profiles, device hashes, IP subnets, and transaction history.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.channels = settings.PAYMENT_CHANNELS
        self.account_history: Dict[str, List[float]] = {}  # account -> list of timestamps
        self.account_last_inflow: Dict[str, float] = {}   # account -> last inbound tx time
        self.account_first_seen: Dict[str, float] = {}    # account -> first seen time
        self.account_devices: Dict[str, set] = {}         # account -> set of device hashes
        self.account_types: Dict[str, str] = {}           # account -> 'merchant', 'mule', 'consumer'

        self._init_pool()

    def _init_pool(self):
        # Pre-generate persistent merchants
        self.merchants = [f"ACC_MERCHANT_{i:04d}" for i in range(50)]
        for m in self.merchants:
            self.account_types[m] = "merchant"
            # Established merchant active for > 50 hours (180,000s)
            base_time = 1_700_000_000.0 - random.uniform(180_000, 500_000)
            self.account_first_seen[m] = base_time
            self.account_devices[m] = {
                f"dev_merchant_{hashlib.sha256(f'merch_{m}_{j}'.encode()).hexdigest()[:8]}"
                for j in range(5)
            }

        # Pre-generate regular consumer accounts
        self.consumers = [f"ACC_USER_{i:05d}" for i in range(1000)]
        for c in self.consumers:
            self.account_types[c] = "consumer"
            self.account_first_seen[c] = 1_700_000_000.0 - random.uniform(10_000, 100_000)
            self.account_devices[c] = {
                f"dev_user_{hashlib.sha256(f'user_{c}'.encode()).hexdigest()[:8]}"
            }

        # Shared device & IP pools for criminal mule rings
        self.syndicate_devices = [
            f"dev_syndicate_{hashlib.md5(f'mule_box_{i}'.encode()).hexdigest()[:8]}"
            for i in range(12)
        ]
        self.syndicate_subnets = [
            f"198.51.100.{i*16}/24" for i in range(6)
        ]

        self.tx_counter = 100000

    def generate_device_hash(self, account: str) -> str:
        """Canvas/WebGL browser hash or hardware UUID."""
        if account in self.account_types and self.account_types[account] == "merchant":
            return random.choice(list(self.account_devices[account]))
        if account in self.account_devices and self.account_devices[account]:
            return next(iter(self.account_devices[account]))
        dev = f"dev_hash_{hashlib.sha256(account.encode()).hexdigest()[:8]}"
        self.account_devices.setdefault(account, set()).add(dev)
        return dev

    def generate_ip_subnet(self, account: str) -> str:
        """Generates /24 CIDR prefix or ASN representation."""
        if account in self.account_types and self.account_types[account] == "merchant":
            return f"104.28.{random.randint(1, 254)}.0/24"
        octets = [172, random.randint(16, 31), random.randint(1, 254), 0]
        return f"{octets[0]}.{octets[1]}.{octets[2]}.0/24"

    def record_transaction(
        self,
        src: str,
        dst: str,
        amount: float,
        timestamp: float,
        device: str,
        subnet: str,
        channel: str,
        cadence_ms: float,
        is_mule: bool,
        typology: str,
    ) -> TransactionEvent:
        self.tx_counter += 1
        tx_id = f"TX_{self.tx_counter:07d}"

        bot_flag = cadence_ms < settings.CADENCE_BOT_THRESHOLD_MS

        if src not in self.account_first_seen:
            self.account_first_seen[src] = timestamp
        if dst not in self.account_first_seen:
            self.account_first_seen[dst] = timestamp

        self.account_history.setdefault(src, []).append(timestamp)
        self.account_history.setdefault(dst, []).append(timestamp)
        self.account_last_inflow[dst] = timestamp

        self.account_devices.setdefault(src, set()).add(device)

        return TransactionEvent(
            tx_id=tx_id,
            src_account=src,
            dst_account=dst,
            amount=amount,
            timestamp=timestamp,
            payment_channel=channel,
            device_fingerprint=device,
            ip_subnet=subnet,
            cadence_ms=cadence_ms,
            cadence_bot_flag=bot_flag,
            is_mule=is_mule,
            typology=typology,
        )

    # ---------------------------------------------------------
    # Typology 1: Money Mule Ring (Cyclic rapid pass-through)
    # ---------------------------------------------------------
    def generate_mule_ring(self, base_time: float, ring_size: int = 5) -> List[TransactionEvent]:
        """
        Generates a tightly coupled mule cycle where funds are received and
        immediately passed through (>85% pass-through within 90s) with shared device/IP telemetry.
        """
        ring_nodes = [f"ACC_MULE_RING_{random.randint(1000, 9999)}_{i}" for i in range(ring_size)]
        shared_device = random.choice(self.syndicate_devices)
        shared_subnet = random.choice(self.syndicate_subnets)
        channel = random.choice(["CRYPTO_BITCOIN", "WIRE", "RTP"])

        events = []
        current_time = base_time
        base_amount = random.uniform(3000.0, 9500.0)

        for i in range(ring_size):
            src = ring_nodes[i]
            dst = ring_nodes[(i + 1) % ring_size]

            # Inflow to first node if start
            if i == 0:
                feeder = random.choice(self.consumers)
                feeder_time = current_time - random.uniform(10.0, 45.0)
                events.append(
                    self.record_transaction(
                        src=feeder,
                        dst=src,
                        amount=base_amount,
                        timestamp=feeder_time,
                        device=self.generate_device_hash(feeder),
                        subnet=self.generate_ip_subnet(feeder),
                        channel=channel,
                        cadence_ms=random.uniform(2000.0, 5000.0),
                        is_mule=False,
                        typology="feeder",
                    )
                )

            # Pass-through: ~92% - 98% passed along within 15-45 seconds
            delta_step = random.uniform(12.0, 38.0)
            current_time += delta_step
            outflow_amount = base_amount * random.uniform(0.92, 0.98)
            base_amount = outflow_amount

            # Automated rapid bot cadence (<800ms)
            cadence = random.uniform(120.0, 650.0)

            events.append(
                self.record_transaction(
                    src=src,
                    dst=dst,
                    amount=outflow_amount,
                    timestamp=current_time,
                    device=shared_device,
                    subnet=shared_subnet,
                    channel=channel,
                    cadence_ms=cadence,
                    is_mule=True,
                    typology="ring",
                )
            )

        return events

    # ---------------------------------------------------------
    # Typology 2: Fan-out Layering (Smurfing)
    # ---------------------------------------------------------
    def generate_smurfing_layer(self, base_time: float, num_mules: int = 12) -> List[TransactionEvent]:
        """
        High-velocity fan-out: single dirty source splits funds across 10-25 intermediate
        mule accounts within seconds, which then aggregate into an exit wallet/account.
        """
        master_src = f"ACC_SMURF_SRC_{random.randint(100, 999)}"
        exit_account = f"ACC_SMURF_EXIT_{random.randint(100, 999)}"
        mule_accounts = [f"ACC_MULE_SMURF_{random.randint(1000, 9999)}_{i}" for i in range(num_mules)]
        shared_device = random.choice(self.syndicate_devices)
        shared_subnet = random.choice(self.syndicate_subnets)

        events = []
        total_dirty = random.uniform(25_000.0, 80_000.0)
        chunk_size = total_dirty / num_mules

        # Step 1: Rapid fan-out from master source
        t = base_time
        for mule in mule_accounts:
            t += random.uniform(0.5, 3.0)  # burst within seconds
            events.append(
                self.record_transaction(
                    src=master_src,
                    dst=mule,
                    amount=chunk_size * random.uniform(0.95, 1.05),
                    timestamp=t,
                    device=shared_device,
                    subnet=shared_subnet,
                    channel="CRYPTO_LIGHTNING",
                    cadence_ms=random.uniform(100.0, 450.0),  # bot-driven
                    is_mule=True,
                    typology="smurfing",
                )
            )

        # Step 2: Mules rapidly pass funds forward to exit aggregator (<60 seconds)
        t_agg = t + random.uniform(10.0, 30.0)
        for mule in mule_accounts:
            t_agg += random.uniform(1.0, 4.0)
            events.append(
                self.record_transaction(
                    src=mule,
                    dst=exit_account,
                    amount=chunk_size * random.uniform(0.88, 0.96),
                    timestamp=t_agg,
                    device=shared_device,
                    subnet=shared_subnet,
                    channel="CRYPTO_BITCOIN",
                    cadence_ms=random.uniform(200.0, 700.0),
                    is_mule=True,
                    typology="smurfing",
                )
            )

        return events

    # ---------------------------------------------------------
    # Typology 3: Peeling Chain (Crypto micro-peels)
    # ---------------------------------------------------------
    def generate_peeling_chain(self, base_time: float, hops: int = 8) -> List[TransactionEvent]:
        """
        High-velocity crypto peeling chain: A large balance transfers through a sequence of
        addresses, peeling off small amounts at each step while the majority continues.
        """
        chain_nodes = [f"ACC_PEEL_{random.randint(1000, 9999)}_{i}" for i in range(hops + 1)]
        shared_device = random.choice(self.syndicate_devices)
        shared_subnet = random.choice(self.syndicate_subnets)

        events = []
        main_balance = random.uniform(15_000.0, 50_000.0)
        current_time = base_time

        for i in range(hops):
            src = chain_nodes[i]
            next_hop = chain_nodes[i + 1]
            peel_dest = f"ACC_COLD_WALLET_{random.randint(1000, 9999)}"

            peel_amt = main_balance * random.uniform(0.04, 0.09)
            pass_amt = main_balance - peel_amt
            main_balance = pass_amt

            current_time += random.uniform(15.0, 45.0)

            # Transaction 1: small peel
            events.append(
                self.record_transaction(
                    src=src,
                    dst=peel_dest,
                    amount=peel_amt,
                    timestamp=current_time,
                    device=shared_device,
                    subnet=shared_subnet,
                    channel="CRYPTO_BITCOIN",
                    cadence_ms=random.uniform(150.0, 550.0),
                    is_mule=True,
                    typology="peeling",
                )
            )

            # Transaction 2: pass-through bulk to next hop
            current_time += random.uniform(2.0, 10.0)
            events.append(
                self.record_transaction(
                    src=src,
                    dst=next_hop,
                    amount=pass_amt,
                    timestamp=current_time,
                    device=shared_device,
                    subnet=shared_subnet,
                    channel="CRYPTO_BITCOIN",
                    cadence_ms=random.uniform(100.0, 400.0),
                    is_mule=True,
                    typology="peeling",
                )
            )

        return events

    # ---------------------------------------------------------
    # Typology 4: Legitimate High-Volume Merchant (Non-Mule)
    # ---------------------------------------------------------
    def generate_merchant_transactions(self, base_time: float, count: int = 15) -> List[TransactionEvent]:
        """
        Legitimate merchant transactions:
          - Established account (>48h history)
          - Multiple independent consumers paying the merchant
          - Distinct devices and subnets
          - Retention of funds (no immediate 90s pass-through)
        """
        merchant = random.choice(self.merchants)
        events = []
        t = base_time

        for _ in range(count):
            payer = random.choice(self.consumers)
            amt = random.uniform(15.0, 750.0)
            t += random.uniform(5.0, 120.0)
            dev = self.generate_device_hash(payer)
            subnet = self.generate_ip_subnet(payer)
            channel = random.choice(["ACH", "WIRE", "SEPA", "RTP"])
            cadence = random.uniform(3500.0, 18000.0)  # human cadence

            events.append(
                self.record_transaction(
                    src=payer,
                    dst=merchant,
                    amount=amt,
                    timestamp=t,
                    device=dev,
                    subnet=subnet,
                    channel=channel,
                    cadence_ms=cadence,
                    is_mule=False,
                    typology="merchant",
                )
            )

        return events

    # ---------------------------------------------------------
    # Typology 5: Standard P2P Non-Mule Transfer
    # ---------------------------------------------------------
    def generate_legit_p2p(self, base_time: float) -> TransactionEvent:
        src = random.choice(self.consumers)
        dst = random.choice(self.consumers)
        while dst == src:
            dst = random.choice(self.consumers)

        amt = random.uniform(10.0, 450.0)
        t = base_time + random.uniform(1.0, 60.0)
        dev = self.generate_device_hash(src)
        subnet = self.generate_ip_subnet(src)
        channel = random.choice(self.channels)
        cadence = random.uniform(4000.0, 15000.0)

        return self.record_transaction(
            src=src,
            dst=dst,
            amount=amt,
            timestamp=t,
            device=dev,
            subnet=subnet,
            channel=channel,
            cadence_ms=cadence,
            is_mule=False,
            typology="legit_p2p",
        )

    # ---------------------------------------------------------
    # Comprehensive Dataset Generator
    # ---------------------------------------------------------
    def generate_batch_dataset(
        self,
        num_transactions: int = 5000,
        mule_ratio: float = 0.05,
    ) -> List[TransactionEvent]:
        """
        Generates a realistic stream containing background legitimate transactions
        interleaved with money mule rings, smurfing bursts, and peeling chains.
        """
        all_events: List[TransactionEvent] = []
        current_time = 1_700_000_000.0

        # Background merchant and P2P
        while len(all_events) < num_transactions:
            roll = random.random()
            if roll < mule_ratio:
                # Inject mule pattern
                mule_type = random.choice(["ring", "smurfing", "peeling"])
                if mule_type == "ring":
                    evs = self.generate_mule_ring(current_time, ring_size=random.randint(4, 7))
                elif mule_type == "smurfing":
                    evs = self.generate_smurfing_layer(current_time, num_mules=random.randint(8, 16))
                else:
                    evs = self.generate_peeling_chain(current_time, hops=random.randint(5, 10))
                all_events.extend(evs)
                current_time += random.uniform(60.0, 180.0)
            elif roll < 0.60:
                evs = self.generate_merchant_transactions(current_time, count=random.randint(3, 8))
                all_events.extend(evs)
                current_time += random.uniform(10.0, 45.0)
            else:
                ev = self.generate_legit_p2p(current_time)
                all_events.append(ev)
                current_time += random.uniform(2.0, 15.0)

        # Sort chronologically
        all_events.sort(key=lambda x: x.timestamp)
        return all_events[:num_transactions]


# Singleton instance
synthetic_generator = SyntheticTelemetryGenerator()
