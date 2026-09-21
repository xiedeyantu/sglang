"""Reset one request's non-online FP32 compressor state across NPU layers."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["req_pool_idx"])
def _clear_dsv4_request_state_kernel(
    state_ptrs,
    req_pool_idx,
    RING_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    layer = tl.program_id(0)
    block = tl.program_id(1)
    state = tl.load(state_ptrs + layer).to(tl.pointer_type(tl.float32))
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Each physical row stores [KV, score]. The initial score must be -inf,
    # including ring rows not written by this request's final partial block.
    value = tl.where(offsets % WIDTH < WIDTH // 2, 0.0, float("-inf"))
    start = req_pool_idx.to(tl.int64) * RING_SIZE * WIDTH
    tl.store(state + start + offsets, value, offsets < RING_SIZE * WIDTH)


def clear_dsv4_request_state(
    state_ptrs: torch.Tensor,
    req_pool_idx: int,
    ring_size: int,
    width: int,
) -> None:
    """Enqueue one reset on the current stream using a persistent pointer table.

    The owner keeps all pointed-to buffers alive. Every buffer is contiguous
    FP32 with the same non-online ring layout; request indices are host ints.
    No D2H, temporary tensor, or stream synchronization is needed here.
    """
    num_layers = state_ptrs.numel()
    if num_layers == 0:
        return
    block_size = 4096
    _clear_dsv4_request_state_kernel[
        (num_layers, triton.cdiv(ring_size * width, block_size))
    ](
        state_ptrs,
        req_pool_idx,
        RING_SIZE=ring_size,
        WIDTH=width,
        BLOCK_SIZE=block_size,
    )
