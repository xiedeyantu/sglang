"""Fuse the fixed-size NPU DSV4 decode compression metadata refresh."""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _store_compressed_positions(
    seq, valid, offsets, dst, BS: tl.constexpr, RATIO: tl.constexpr
):
    keep = valid & (seq % RATIO == 0)
    ordinal = tl.cumsum(keep.to(tl.int32), 0) - 1
    count = tl.sum(keep.to(tl.int32), 0)
    # Selected requests own [0, count), in their original order. Tail stores
    # own [count, BS): disjoint writes, with no zero-then-scatter race.
    tl.store(dst + ordinal, seq.to(tl.int64) - RATIO, mask=keep)
    tl.store(dst + offsets, 0, mask=(offsets >= count) & (offsets < BS))


@triton.jit(do_not_specialize=["n_c4", "n_c128"])
def _refresh_dsv4_decode_metadata_kernel(
    seq_lens,
    c4_src,
    c128_src,
    c4_loc,
    c128_loc,
    c4_positions,
    c128_positions,
    start_pos,
    seqused,
    n_c4,
    n_c128,
    BS: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    C4_STRIDE: tl.constexpr,
    C128_STRIDE: tl.constexpr,
    HAS_C4: tl.constexpr,
    HAS_C128: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    seq = tl.load(seq_lens + offsets * SEQ_STRIDE, mask=offsets < BS, other=0)
    valid = (offsets < BS) & (seq > 0)
    tl.store(start_pos + offsets, tl.maximum(seq - 1, 0), mask=offsets < BS)
    tl.store(seqused + offsets, valid.to(tl.int32), mask=offsets < BS)

    if HAS_C4:
        loc = tl.load(c4_src + offsets * C4_STRIDE, mask=offsets < n_c4, other=0)
        tl.store(c4_loc + offsets, loc, mask=offsets < BS)
        _store_compressed_positions(seq, valid, offsets, c4_positions, BS, 4)
    if HAS_C128:
        loc = tl.load(c128_src + offsets * C128_STRIDE, mask=offsets < n_c128, other=0)
        tl.store(c128_loc + offsets, loc, mask=offsets < BS)
        _store_compressed_positions(seq, valid, offsets, c128_positions, BS, 128)


def refresh_dsv4_decode_metadata(
    seq_lens: torch.Tensor,
    *,
    c4_src: Optional[torch.Tensor],
    c128_src: Optional[torch.Tensor],
    c4_loc: torch.Tensor,
    c128_loc: torch.Tensor,
    c4_positions: torch.Tensor,
    c128_positions: torch.Tensor,
    start_pos: torch.Tensor,
    seqused: torch.Tensor,
    has_c4: bool,
    has_c128: bool,
) -> None:
    """Update persistent decode buffers on the current stream, without scratch.

    Decode has one output slot per graph request in all six contiguous output
    vectors. Zero sequence lengths represent graph padding/idle rows. Location
    sources may be missing, empty, strided or a different integer dtype; only
    enabled compression ratios are written. Counts are runtime scalars so a
    C4/C128 boundary does not trigger count-specific JIT compilation.
    """
    bs = seq_lens.numel()
    assert seq_lens.ndim == 1
    for output in (c4_loc, c128_loc, c4_positions, c128_positions, start_pos, seqused):
        assert output.ndim == 1 and output.numel() == bs and output.is_contiguous(), (
            "fused decode metadata requires one contiguous output slot per request"
        )
    n_c4 = c4_src.numel() if has_c4 and c4_src is not None else 0
    n_c128 = c128_src.numel() if has_c128 and c128_src is not None else 0
    assert n_c4 <= bs and n_c128 <= bs, (
        f"graph replay 1D metadata overflow: c4={n_c4}, c128={n_c128}, dst={bs}"
    )
    if bs == 0:
        return
    # Masked-out loads use a valid device pointer even for empty source tensors.
    # No placeholder tensor or dtype-conversion allocation is needed.
    c4_src = c4_src if n_c4 else c4_loc
    c128_src = c128_src if n_c128 else c128_loc
    _refresh_dsv4_decode_metadata_kernel[(1,)](
        seq_lens,
        c4_src,
        c128_src,
        c4_loc,
        c128_loc,
        c4_positions,
        c128_positions,
        start_pos,
        seqused,
        n_c4,
        n_c128,
        BS=bs,
        SEQ_STRIDE=seq_lens.stride(0),
        C4_STRIDE=c4_src.stride(0),
        C128_STRIDE=c128_src.stride(0),
        HAS_C4=has_c4,
        HAS_C128=has_c128,
        BLOCK_SIZE=triton.next_power_of_2(bs),
    )
