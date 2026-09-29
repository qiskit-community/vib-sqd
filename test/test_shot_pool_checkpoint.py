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

"""Streaming NDJSON shot-pool checkpoint: round-trip correctness against a
naive Counter merge, feeding straight into recover_direct_onehot_distribution,
and crash-safety (a truncated trailing line must not lose prior batches).
"""

from collections import Counter

import pytest

from vib_sqd.sqd.shot_pool_checkpoint import (
    append_batch,
    iter_batches,
    load_pooled_counts,
    load_run_metadata,
)
from vib_sqd.sqd.vibrational_recovery import (
    initial_uniform_modal_occupancies,
    recover_direct_onehot_distribution,
)

BATCHES = [
    ("job_a", {"000100": 10, "010001": 5, "111111": 1}),  # some invalid one-hot too
    ("job_b", {"000100": 7, "001000": 3}),  # overlaps batch 1's key
    ("job_c", {"100010": 2.0, "010001": 4.0}),  # float counts + more overlap
]


def _write_all(ckpt_path):
    for i, (job_id, counts) in enumerate(BATCHES):
        append_batch(ckpt_path, batch_index=i, job_id=job_id, counts=counts)


def test_load_pooled_counts_matches_naive_counter_merge(tmp_path):
    ckpt = tmp_path / "pool.ndjson"
    _write_all(ckpt)

    expected = Counter()
    for _, counts in BATCHES:
        for k, v in counts.items():
            expected[k] += float(v)

    pooled = load_pooled_counts(ckpt)
    assert pooled == dict(expected)


def test_append_is_incremental_not_full_rewrite(tmp_path):
    """Each append must only grow the file by roughly that batch's own size --
    not re-serialize everything written so far."""
    ckpt = tmp_path / "pool.ndjson"
    sizes = []
    for i, (job_id, counts) in enumerate(BATCHES):
        append_batch(ckpt, batch_index=i, job_id=job_id, counts=counts)
        sizes.append(ckpt.stat().st_size)
    # size after batch 3 should be roughly the sum of the three batches' own
    # payload sizes, not e.g. quadratic growth from re-writing prior batches.
    deltas = [sizes[0]] + [sizes[i] - sizes[i - 1] for i in range(1, len(sizes))]
    # each individual batch's on-disk delta should be small and comparable
    # (not growing with how many batches came before it)
    assert max(deltas) < 300  # generous bound; these are tiny synthetic batches


def test_load_run_metadata(tmp_path):
    ckpt = tmp_path / "pool.ndjson"
    _write_all(ckpt)
    meta = load_run_metadata(ckpt)
    assert meta["batches_done"] == 3
    assert meta["job_ids"] == ["job_a", "job_b", "job_c"]


def test_pooled_counts_feed_recover_direct_onehot_distribution(tmp_path):
    """The concrete contract: the loaded dict must be usable exactly as
    recover_direct_onehot_distribution expects, and produce a normalized
    (sums to 1.0) recovered distribution."""
    ckpt = tmp_path / "pool.ndjson"
    _write_all(ckpt)
    pooled = load_pooled_counts(ckpt)

    n_modes, n_modals, vmax = 3, 2, 1
    occ = initial_uniform_modal_occupancies(n_modes, n_modals, vmax)
    rec = recover_direct_onehot_distribution(
        pooled,
        avg_modal_occupancies=occ,
        n_modes=n_modes,
        n_modals=n_modals,
        vmax=vmax,
        seed=0,
        randomize=True,
    )
    assert isinstance(rec, dict)
    assert sum(rec.values()) == pytest.approx(1.0)


def test_truncated_trailing_line_does_not_lose_prior_batches(tmp_path):
    """Simulate a crash mid-write of the last batch: the file ends with a
    partial JSON line. Prior complete batches must still load correctly."""
    ckpt = tmp_path / "pool.ndjson"
    _write_all(ckpt)
    with open(ckpt, "a") as f:
        f.write('{"batch": 3, "job_id": "job_d", "counts": {"000')  # truncated

    pooled = load_pooled_counts(ckpt)
    expected = Counter()
    for _, counts in BATCHES:
        for k, v in counts.items():
            expected[k] += float(v)
    assert pooled == dict(expected)


def test_corrupt_interior_line_raises(tmp_path):
    """A corrupt line that is NOT the last one is a real bug and must raise,
    not be silently swallowed like a crash-truncated final line."""
    ckpt = tmp_path / "pool.ndjson"
    append_batch(ckpt, 0, "job_a", {"0001": 1})
    with open(ckpt, "a") as f:
        f.write("not valid json at all\n")
    append_batch(ckpt, 2, "job_c", {"0010": 1})

    with pytest.raises(Exception):
        list(iter_batches(ckpt))
