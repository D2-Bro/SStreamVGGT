#!/usr/bin/env python3
"""Smoke tests for token-overlay event dumps."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from streamvggt.utils.cache_analysis import (  # noqa: E402
    TokenOverlayDumpConfig,
    dump_token_overlay_event,
)


def check_head_shared_metadata_dump() -> None:
    tmp = tempfile.mkdtemp(prefix="token_overlay_dump_")
    try:
        config = TokenOverlayDumpConfig(tmp)
        metadata = SimpleNamespace(
            frame_ids=torch.tensor([[0, 0, 1, 1, 2, 2]]),
            token_indices=torch.tensor([[0, 1, 0, 1, 0, 1]]),
        )
        dump_token_overlay_event(
            config,
            kept_candidate_indices=torch.tensor([[[0, 2]]]),
            policy_scores=torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            metadata=metadata,
            layer_id=5,
            step_idx=2,
            cache_budget=4,
            num_anchor_tokens=2,
            tokens_per_frame=2,
            eviction_policy="svd_leverage",
            leverage_granularity="layer",
            selection_granularity="layer",
        )
        event_path = os.path.join(tmp, "events", "step_000002_layer_05_head_layer_shared_batch_00.pt")
        if not os.path.exists(event_path):
            raise AssertionError("head-shared token overlay event was not written")
        payload = torch.load(event_path, map_location="cpu", weights_only=False)
        if payload["candidate_frame_ids"].tolist() != [1, 1, 2, 2]:
            raise AssertionError(f"unexpected candidate frame ids: {payload['candidate_frame_ids']}")
        if payload["evicted_frame_ids"].tolist() != [1, 2]:
            raise AssertionError(f"unexpected evicted frame ids: {payload['evicted_frame_ids']}")
        if payload["evicted_token_indices"].tolist() != [1, 1]:
            raise AssertionError(f"unexpected evicted token indices: {payload['evicted_token_indices']}")
    finally:
        shutil.rmtree(tmp)


def main() -> None:
    check_head_shared_metadata_dump()
    print("token overlay dump smoke tests passed")


if __name__ == "__main__":
    main()
