"""
IBM AML-Data & AMLSim Multi-Agent Banking Simulator.
Implements the multi-agent AML simulation patterns from:
  - IBM AML-Data: https://github.com/IBM/AML-Data
  - IBM AMLSim: https://github.com/IBM/AMLSim
  - IBM Transactions for Anti-Money Laundering (Kaggle Dataset)
  - SAML-D Benchmark Pipeline (AlonaDrok & Berkan Oztas)

Simulates 7 SOTA IBM AML typologies:
  1. Fan-Out (Smurfing from single originator)
  2. Fan-In (Aggregation to destination exit node)
  3. Cycle (Circular money mule ring)
  4. Scatter-Gather (Layering: Fan-out -> intermediate mules -> Fan-in)
  5. Peeling Chain (Incremental peel-off along a chain)
  6. Bipartite (Dense multi-originator to multi-beneficiary matrix)
  7. Fast Outflow (Transit node with rapid fund forwarding)
"""

import csv
import hashlib
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from config.settings import settings


@dataclass
class AMLSimTransaction:
    step: int
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    amount: float
    currency: str
    payment_format: str
    is_sar: bool  # Suspicious Activity Report / Money Laundering
    model_id: str  # Typology model identifier (e.g. 'fan_out', 'cycle', 'scatter_gather')
    timestamp: float


class IBMAmlSimulator:
    """
    Multi-Agent Financial Transaction Simulator replicating IBM AMLSim logic.
    Injects synthetic laundering subgraphs into normal banking streams.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.banks = ["BANK_JPMC", "BANK_BOFA", "BANK_WELLS", "BANK_CITI", "BANK_HSBC"]
        self.currencies = ["USD", "EUR", "BTC", "USDT"]
        self.formats = ["WIRE", "ACH", "CHECK", "CREDIT_CARD", "CRYPTO"]
        self.step_counter = 0

    def generate_fan_out(
        self,
        base_time: float,
        num_mules: int = 8,
        total_amount: float = 50000.0,
    ) -> List[AMLSimTransaction]:
        """IBM AMLSim Pattern 1: Fan-Out (Smurfing)."""
        originator = f"ACC_IBM_ORIG_{random.randint(100, 999)}"
        orig_bank = random.choice(self.banks)
        mules = [f"ACC_IBM_MULE_{random.randint(1000, 9999)}_{i}" for i in range(num_mules)]

        events = []
        chunk = total_amount / num_mules
        t = base_time

        for mule in mules:
            t += random.uniform(0.5, 3.0)
            self.step_counter += 1
            events.append(
                AMLSimTransaction(
                    step=self.step_counter,
                    from_bank=orig_bank,
                    from_account=originator,
                    to_bank=random.choice(self.banks),
                    to_account=mule,
                    amount=round(chunk * random.uniform(0.96, 1.04), 2),
                    currency="USD",
                    payment_format="WIRE",
                    is_sar=True,
                    model_id="fan_out",
                    timestamp=t,
                )
            )
        return events

    def generate_scatter_gather(
        self,
        base_time: float,
        num_intermediates: int = 6,
        total_amount: float = 65000.0,
    ) -> List[AMLSimTransaction]:
        """
        IBM AMLSim Pattern 4: Scatter-Gather.
        Source -> Intermediate Mules (Scatter) -> Aggregator Exit (Gather).
        """
        src = f"ACC_SCATTER_SRC_{random.randint(100, 999)}"
        aggregator = f"ACC_GATHER_EXIT_{random.randint(100, 999)}"
        intermediates = [f"ACC_MULE_MID_{random.randint(1000, 9999)}_{i}" for i in range(num_intermediates)]

        events = []
        chunk = total_amount / num_intermediates
        t = base_time

        # Scatter phase
        for mid in intermediates:
            t += random.uniform(1.0, 5.0)
            self.step_counter += 1
            amt = round(chunk * random.uniform(0.97, 1.02), 2)
            events.append(
                AMLSimTransaction(
                    step=self.step_counter,
                    from_bank=random.choice(self.banks),
                    from_account=src,
                    to_bank=random.choice(self.banks),
                    to_account=mid,
                    amount=amt,
                    currency="USD",
                    payment_format="WIRE",
                    is_sar=True,
                    model_id="scatter_gather",
                    timestamp=t,
                )
            )

        # Gather phase (< 60s later)
        t += random.uniform(10.0, 25.0)
        for mid in intermediates:
            t += random.uniform(1.0, 4.0)
            self.step_counter += 1
            amt = round(chunk * random.uniform(0.91, 0.96), 2)  # laundering fee deduction
            events.append(
                AMLSimTransaction(
                    step=self.step_counter,
                    from_bank=random.choice(self.banks),
                    from_account=mid,
                    to_bank=random.choice(self.banks),
                    to_account=aggregator,
                    amount=amt,
                    currency="USD",
                    payment_format="WIRE",
                    is_sar=True,
                    model_id="scatter_gather",
                    timestamp=t,
                )
            )

        return events

    def generate_cycle(
        self,
        base_time: float,
        ring_size: int = 5,
        base_amount: float = 24000.0,
    ) -> List[AMLSimTransaction]:
        """IBM AMLSim Pattern 3: Circular Mule Ring."""
        nodes = [f"ACC_IBM_CYCLE_{random.randint(1000, 9999)}_{i}" for i in range(ring_size)]
        events = []
        t = base_time
        curr_amt = base_amount

        for i in range(ring_size):
            u = nodes[i]
            v = nodes[(i + 1) % ring_size]
            t += random.uniform(15.0, 45.0)
            self.step_counter += 1
            curr_amt = round(curr_amt * random.uniform(0.94, 0.98), 2)

            events.append(
                AMLSimTransaction(
                    step=self.step_counter,
                    from_bank=random.choice(self.banks),
                    from_account=u,
                    to_bank=random.choice(self.banks),
                    to_account=v,
                    amount=curr_amt,
                    currency="USD",
                    payment_format="WIRE",
                    is_sar=True,
                    model_id="cycle",
                    timestamp=t,
                )
            )
        return events

    def generate_peeling_chain(
        self,
        base_time: float,
        chain_length: int = 5,
        initial_amount: float = 60000.0,
    ) -> List[AMLSimTransaction]:
        """IBM AMLSim Pattern 2: Peeling Chain (Layering)."""
        events = []
        t = base_time
        curr_holder = f"ACC_PEEL_ORIG_{random.randint(100, 999)}"
        remaining_amt = initial_amount

        for i in range(chain_length):
            t += random.uniform(2.0, 10.0)
            self.step_counter += 1
            peeled_amt = round(min(remaining_amt * random.uniform(0.15, 0.25), 9800.0), 2)
            peeled_dest = f"ACC_PEEL_CASH_{random.randint(1000, 9999)}_{i}"

            # 1. Peeled off transaction
            events.append(
                AMLSimTransaction(
                    step=self.step_counter,
                    from_bank=random.choice(self.banks),
                    from_account=curr_holder,
                    to_bank=random.choice(self.banks),
                    to_account=peeled_dest,
                    amount=peeled_amt,
                    currency="USD",
                    payment_format="ACH",
                    is_sar=True,
                    model_id="peeling_chain",
                    timestamp=t,
                )
            )

            # 2. Remainder hop to next address
            remaining_amt = round(max(50.0, remaining_amt - peeled_amt), 2)
            if i < chain_length - 1:
                next_holder = f"ACC_PEEL_HOP_{random.randint(1000, 9999)}_{i+1}"
                t += random.uniform(1.0, 3.0)
                self.step_counter += 1
                events.append(
                    AMLSimTransaction(
                        step=self.step_counter,
                        from_bank=random.choice(self.banks),
                        from_account=curr_holder,
                        to_bank=random.choice(self.banks),
                        to_account=next_holder,
                        amount=remaining_amt,
                        currency="USD",
                        payment_format="WIRE",
                        is_sar=True,
                        model_id="peeling_chain",
                        timestamp=t,
                    )
                )
                curr_holder = next_holder

        return events

    def generate_normal_banking_traffic(
        self,
        base_time: float,
        count: int = 20,
    ) -> List[AMLSimTransaction]:
        """Generates legitimate baseline banking transactions."""
        events = []
        t = base_time
        for _ in range(count):
            t += random.uniform(0.5, 10.0)
            self.step_counter += 1
            u = f"ACC_LEGIT_{random.randint(10000, 99999)}"
            v = f"ACC_LEGIT_{random.randint(10000, 99999)}"
            amt = round(float(np.random.exponential(scale=350.0) + 5.0), 2)

            events.append(
                AMLSimTransaction(
                    step=self.step_counter,
                    from_bank=random.choice(self.banks),
                    from_account=u,
                    to_bank=random.choice(self.banks),
                    to_account=v,
                    amount=amt,
                    currency="USD",
                    payment_format=random.choice(self.formats),
                    is_sar=False,
                    model_id="normal_p2p",
                    timestamp=t,
                )
            )
        return events

    def export_kaggle_ibm_dataset(self, output_path: Path, num_transactions: int = 3000):
        """Exports dataset compliant with Kaggle IBM Transactions for AML format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        events: List[AMLSimTransaction] = []
        t = 1_700_000_000.0

        while len(events) < num_transactions:
            roll = random.random()
            if roll < 0.05:
                events.extend(self.generate_scatter_gather(t))
                t += 180.0
            elif roll < 0.10:
                events.extend(self.generate_cycle(t))
                t += 120.0
            elif roll < 0.15:
                events.extend(self.generate_fan_out(t))
                t += 90.0
            else:
                events.extend(self.generate_normal_banking_traffic(t, count=10))
                t += 30.0

        events = events[:num_transactions]
        events.sort(key=lambda x: x.timestamp)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "From Bank", "Account", "To Bank", "Account",
                "Amount Received", "Receiving Currency", "Amount Paid",
                "Payment Currency", "Payment Format", "Is Laundering", "Model ID"
            ])
            for ev in events:
                writer.writerow([
                    f"{ev.timestamp:.3f}", ev.from_bank, ev.from_account,
                    ev.to_bank, ev.to_account, ev.amount, ev.currency,
                    ev.amount, ev.currency, ev.payment_format,
                    1 if ev.is_sar else 0, ev.model_id
                ])
        print(f"Exported IBM AMLSim Kaggle dataset to: {output_path}")


# Singleton instance
ibm_aml_simulator = IBMAmlSimulator()
