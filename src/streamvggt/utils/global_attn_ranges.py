from typing import List, Optional, Sequence, Tuple

GlobalAttnIdxRange = Tuple[int, Optional[int]]


def parse_global_attn_idx_ranges(
    range_str: str,
    num_global_blocks: Optional[int] = None,
) -> List[GlobalAttnIdxRange]:
    """Parse half-open global-attention index ranges.

    Examples:
        "9:" -> [(9, None)]
        "9:20" -> [(9, 20)]
        ":9" -> [(0, 9)]
        "12" -> [(12, 13)]
        "6:10,14:20" -> [(6, 10), (14, 20)]
    """
    if range_str is None:
        raise ValueError("global attention range string cannot be None")
    if not isinstance(range_str, str):
        raise ValueError(f"global attention ranges must be a string, got {type(range_str).__name__}")

    range_str = range_str.strip()
    if not range_str:
        raise ValueError("global attention range string cannot be empty")

    ranges: List[GlobalAttnIdxRange] = []
    for part in range_str.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty range in global attention ranges: {range_str!r}")

        if ":" in part:
            if part.count(":") != 1:
                raise ValueError(f"invalid global attention range {part!r}: expected at most one ':'")
            start_str, end_str = part.split(":")
            start = 0 if start_str == "" else _parse_non_negative_int(start_str, part)
            end = None if end_str == "" else _parse_non_negative_int(end_str, part)
        else:
            start = _parse_non_negative_int(part, part)
            end = start + 1

        if end is not None and start >= end:
            raise ValueError(f"invalid global attention range {part!r}: start must be < end")
        if num_global_blocks is not None:
            if num_global_blocks < 0:
                raise ValueError(f"num_global_blocks must be >= 0, got {num_global_blocks}")
            if start >= num_global_blocks:
                raise ValueError(
                    f"global attention range {part!r} starts at {start}, "
                    f"but only {num_global_blocks} global blocks exist"
                )
            if end is not None and end > num_global_blocks:
                raise ValueError(
                    f"global attention range {part!r} ends at {end}, "
                    f"but only {num_global_blocks} global blocks exist"
                )
        ranges.append((start, end))

    return ranges


def is_global_idx_enabled(
    global_idx: int,
    ranges: Sequence[GlobalAttnIdxRange],
) -> bool:
    """Return whether a zero-based global attention index is enabled."""
    if global_idx < 0:
        raise ValueError(f"global_idx must be >= 0, got {global_idx}")
    for start, end in ranges:
        if global_idx >= start and (end is None or global_idx < end):
            return True
    return False


def _parse_non_negative_int(value: str, part: str) -> int:
    if value.strip() != value or value == "":
        raise ValueError(f"invalid global attention range {part!r}: empty or padded integer")
    if not value.isdigit():
        raise ValueError(f"invalid global attention range {part!r}: expected non-negative integers")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"invalid global attention range {part!r}: indices must be non-negative")
    return parsed
