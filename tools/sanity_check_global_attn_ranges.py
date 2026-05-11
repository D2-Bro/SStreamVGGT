#!/usr/bin/env python
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from streamvggt.models.aggregator import Aggregator
from streamvggt.utils.global_attn_ranges import (
    is_global_idx_enabled,
    parse_global_attn_idx_ranges,
)


def main():
    torch.manual_seed(0)
    check_parser()
    check_budget_redistribution()
    check_streaming_modes()
    print("global attention range sanity checks passed")


def check_parser():
    expected = {
        "9:": [(9, None)],
        "9:20": [(9, 20)],
        ":9": [(0, 9)],
        "12": [(12, 13)],
        "6:10,14:20": [(6, 10), (14, 20)],
    }
    for text, parsed in expected.items():
        assert parse_global_attn_idx_ranges(text) == parsed

    for bad in ("", "1::2", "-1", "3:3", "4:2", "1,,2"):
        try:
            parse_global_attn_idx_ranges(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected parser failure for {bad!r}")


def check_streaming_modes():
    depth = 24
    cases = [
        (None, set(range(depth))),
        ("9:", set(range(9, depth))),
        ("9:20", set(range(9, 20))),
        ("6:10,14:20", set(range(6, 10)) | set(range(14, 20))),
        ("12", {12}),
    ]

    for ranges, expected_enabled in cases:
        agg, past_key_values = run_two_streaming_steps(ranges, depth=depth)
        global_trace = [
            row for row in agg.last_global_attn_debug_trace
            if row["original_attention_type"] == "global"
        ]
        assert len(global_trace) == depth

        enabled = {row["global_idx"] for row in global_trace if row["global_enabled"]}
        assert enabled == expected_enabled, f"{ranges}: expected {expected_enabled}, got {enabled}"

        for row in global_trace:
            idx = row["global_idx"]
            should_enable = idx in expected_enabled
            assert row["global_enabled"] is should_enable
            assert row["g2f_conversion"] is (not should_enable)
            assert row["kv_read"] is should_enable
            assert row["kv_write"] is should_enable
            if should_enable:
                assert past_key_values[idx] is not None
            else:
                assert past_key_values[idx] is None

    ranges = parse_global_attn_idx_ranges("6:10,14:20")
    assert is_global_idx_enabled(6, ranges)
    assert is_global_idx_enabled(19, ranges)
    assert not is_global_idx_enabled(13, ranges)


def check_budget_redistribution():
    depth = 24
    total_budget = 2400
    agg = make_tiny_aggregator(depth)
    baseline_budgets = agg._calculate_dynamic_budgets(total_budget)
    range_budgets = agg._calculate_dynamic_budgets(
        total_budget,
        enabled_global_idx_ranges=parse_global_attn_idx_ranges("9:", num_global_blocks=depth),
    )

    assert torch.all(range_budgets[:9] == 0)
    assert torch.all(range_budgets[9:] > baseline_budgets[9:])
    assert int(range_budgets.sum().item()) == total_budget


def run_two_streaming_steps(ranges, depth):
    agg = make_tiny_aggregator(depth)
    agg.eval()
    past_key_values = [None] * agg.depth

    with torch.no_grad():
        for step in range(2):
            images = torch.rand(1, 1, 3, 16, 16)
            output_list, patch_start_idx, past_key_values = agg(
                images,
                past_key_values=past_key_values,
                use_cache=True,
                past_frame_idx=step,
                total_budget=1000,
                global_attn_idx_ranges=ranges,
                global_attn_debug=False,
            )
            assert len(output_list) == depth
            assert patch_start_idx == 3

    return agg, past_key_values


def make_tiny_aggregator(depth):
    return Aggregator(
        img_size=16,
        patch_size=8,
        embed_dim=32,
        depth=depth,
        num_heads=4,
        mlp_ratio=1.0,
        num_register_tokens=2,
        patch_embed="conv",
        rope_freq=100,
    )


if __name__ == "__main__":
    main()
