# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Streaming (NDJSON) checkpoint for pooled hardware-shot counts.

The old approach (a single flat JSON re-serializing the FULL cumulative pool
after every batch) is O(cumulative pool size) per write -- fine at 5M shots/
365MB, but a 10-50x larger 100-qubit pool would push every batch's write to
multi-GB. This module instead appends ONE line per batch (that batch's raw
counts only, never the running total) -- O(batch size) per write, independent
of how large the pool has already grown.

``recover_direct_onehot_distribution`` (vibrational_recovery.py) only ever
needs a single read-only ``dict[str, int|float]`` in memory at call time --
``load_pooled_counts`` reconstructs exactly that shape by merging batches with
a ``Counter``, reading one line's payload at a time.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path


def append_batch(
    ckpt_path: Path,
    batch_index: int,
    job_id: str,
    counts: dict[str, int | float],
    meta: dict | None = None,
) -> None:
    """Append one batch's raw (unmerged) counts as a single NDJSON line.
    O(batch size), NOT O(cumulative pool size) -- never rewrites prior lines."""
    record = {"batch": batch_index, "job_id": job_id, "counts": counts}
    if meta:
        record["meta"] = meta
    with open(ckpt_path, "a") as f:
        f.write(json.dumps(record, default=float))
        f.write("\n")


def iter_batches(ckpt_path: Path) -> Iterator[dict]:
    """Yield each batch record in file order, streaming line-by-line (never
    holds more than one batch's counts in memory at a time).

    A truncated/corrupt trailing line (e.g. a crash mid-write) is silently
    skipped -- at most the in-flight batch is lost, not the whole pool. A
    corrupt line that is NOT the last line is a real bug and re-raises.
    """
    with open(ckpt_path, "r") as f:
        lines = f.readlines()
    n = len(lines)
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if i == n - 1:
                continue  # tolerate a truncated final line from a crash
            raise


def load_pooled_counts(ckpt_path: Path) -> dict[str, float]:
    """Merge all batches into the dict[str, float] shape
    ``recover_direct_onehot_distribution`` requires. Only the running merged
    Counter is held in memory (the same footprint the old in-process ``pooled``
    dict already had) -- individual batch payloads are read and discarded one
    at a time, so this never holds two full copies of the pool at once."""
    pooled: Counter[str] = Counter()
    for rec in iter_batches(ckpt_path):
        for bitstring, count in rec["counts"].items():
            pooled[bitstring] += float(count)
    return dict(pooled)


def load_run_metadata(ckpt_path: Path) -> dict:
    """Cheap batches_done/job_ids inspection without merging any counts."""
    job_ids = []
    n_batches = 0
    for rec in iter_batches(ckpt_path):
        n_batches += 1
        job_ids.append(rec["job_id"])
    return {"batches_done": n_batches, "job_ids": job_ids}
