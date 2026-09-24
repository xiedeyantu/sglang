"""Compare Q absorption weight layouts on one Ascend NPU.

Run with this checkout installed (or PYTHONPATH=python):
    python test/manual/ascend/bench_q_absorb_layout.py --profile-dir /tmp/q-absorb

Defaults model the GLM5.2 TP8 profile: 78 distinct layer weights, 8 local heads,
Q-nope width 192, RoPE width 64, and KV rank 512. Both variants consume a slice
of the full Q tensor. Weight preparation is outside capture and timing.

The candidate uses the real post_load_weights implementation. Validate eager
and graph results against CPU FP32, then reload a weight through that loader
and verify that the captured graph observes it without changing its address.
Each timing sample measures all layers per replay, not a full model step.
Optional traces contain one replay per variant, for checking CANN transposes.
Use --op mla to check the ordinary torch.bmm consumer as well.
"""

import argparse
import json
import math
import statistics
from types import SimpleNamespace

import torch


def _capture(run):
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            run()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        outputs = run()
    return graph, outputs


def _time_graph(graph, iterations):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _check(actual, reference):
    torch.testing.assert_close(actual.cpu().float(), reference, rtol=0.016, atol=0.016)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--layers", type=int, default=78)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--qk-nope-dim", type=int, default=192)
    parser.add_argument("--qk-rope-dim", type=int, default=64)
    parser.add_argument("--kv-rank", type=int, default=512)
    parser.add_argument("--v-dim", type=int, default=128)
    parser.add_argument("--op", choices=["dsa", "mla"], default="dsa")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--profile-dir")
    args = parser.parse_args()
    if (
        min(
            *args.batch_sizes,
            args.layers,
            args.heads,
            args.qk_nope_dim,
            args.qk_rope_dim,
            args.kv_rank,
            args.v_dim,
            args.iterations,
            args.rounds,
        )
        < 1
    ):
        parser.error("dimensions and iteration counts must be positive")

    import torch_npu

    from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
        DeepseekV2WeightLoaderMixin,
    )

    torch.npu.set_device(args.device)
    torch.manual_seed(42)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "device": torch.npu.get_device_name(),
                "args": vars(args),
            }
        ),
        flush=True,
    )
    attentions = [
        SimpleNamespace(
            kv_b_proj=SimpleNamespace(
                weight=torch.randn(
                    args.heads * (args.qk_nope_dim + args.v_dim),
                    args.kv_rank,
                    device=args.device,
                    dtype=torch.bfloat16,
                )
                / math.sqrt(args.qk_nope_dim)
            ),
            qk_nope_head_dim=args.qk_nope_dim,
            v_head_dim=args.v_dim,
            w_kc=None,
            w_vc=None,
            w_scale=None,
        )
        for _ in range(args.layers)
    ]
    loader = SimpleNamespace(
        model=SimpleNamespace(
            start_layer=0,
            end_layer=args.layers,
            layers=[SimpleNamespace(self_attn=a) for a in attentions],
        ),
        quant_config=None,
        config=SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"]),
    )
    post_load = DeepseekV2WeightLoaderMixin.post_load_weights
    post_load(loader)
    if not all(a.w_kc.is_contiguous() for a in attentions):
        raise RuntimeError("Run with the updated NPU weight loader from this checkout")
    weights = {
        "original": [
            a.w_kc.transpose(1, 2).contiguous().transpose(1, 2) for a in attentions
        ],
        "prepacked": [a.w_kc for a in attentions],
    }
    # Build the reference from checkpoint weights, independently of w_kc.
    cpu_weights = [
        a.kv_b_proj.weight.cpu()
        .float()
        .view(args.heads, args.qk_nope_dim + args.v_dim, args.kv_rank)[
            :, : args.qk_nope_dim
        ]
        for a in attentions
    ]

    def absorb(q, w):
        q_nope, _ = q.split([args.qk_nope_dim, args.qk_rope_dim], dim=-1)
        if args.op == "mla":
            return torch.bmm(q_nope.transpose(0, 1), w).transpose(0, 1)
        return torch_npu.npu_transpose_batchmatmul(
            q_nope, w, perm_x1=(1, 0, 2), perm_x2=(0, 1, 2), perm_y=(1, 0, 2)
        )

    for tokens in args.batch_sizes:
        inputs = [
            torch.randn(
                tokens,
                args.heads,
                args.qk_nope_dim + args.qk_rope_dim,
                device=args.device,
                dtype=torch.bfloat16,
            )
            for _ in attentions
        ]
        references = [
            torch.bmm(
                q.cpu().float()[..., : args.qk_nope_dim].transpose(0, 1), w
            ).transpose(0, 1)
            for q, w in zip(inputs, cpu_weights)
        ]
        graphs, outputs = {}, {}
        for name, tensors in weights.items():

            def run(tensors=tensors):
                return [absorb(q, w) for q, w in zip(inputs, tensors)]

            for actual, reference in zip(run(), references):
                _check(actual, reference)
            graphs[name], outputs[name] = _capture(run)
            graphs[name].replay()
            for actual, reference in zip(outputs[name], references):
                _check(actual, reference)

        # Reload only layer 0, through the same path as partial weight updates.
        attn = attentions[0]
        pointers = (attn.w_kc.data_ptr(), attn.w_vc.data_ptr())
        value_weight = attn.w_vc.clone()
        attn.kv_b_proj.weight.neg_()
        post_load(loader, weight_names=["model.layers.0.self_attn.kv_b_proj.weight"])
        assert pointers == (attn.w_kc.data_ptr(), attn.w_vc.data_ptr())
        torch.testing.assert_close(attn.w_vc, -value_weight, rtol=0, atol=0)
        weights["original"][0].copy_(attn.w_kc)
        for name, graph in graphs.items():
            graph.replay()
            for i, (actual, reference) in enumerate(zip(outputs[name], references)):
                _check(actual, -reference if i == 0 else reference)
        attn.kv_b_proj.weight.neg_()
        post_load(loader, weight_names=["model.layers.0.self_attn.kv_b_proj.weight"])
        weights["original"][0].copy_(attn.w_kc)

        for name, graph in graphs.items():
            for _ in range(5):
                graph.replay()
            for actual, reference in zip(outputs[name], references):
                _check(actual, reference)
        samples = {name: [] for name in graphs}
        names = list(graphs)
        for round_id in range(args.rounds):
            # Alternate order to reduce systematic timing bias.
            offset = round_id % len(names)
            for name in names[offset:] + names[:offset]:
                samples[name].append(_time_graph(graphs[name], args.iterations))
        baseline = statistics.median(samples["original"])
        for name, times in samples.items():
            elapsed = statistics.median(times)
            print(
                json.dumps(
                    {
                        "batch_size": tokens,
                        "variant": name,
                        "correctness_and_reload": "passed",
                        "weight_stride": list(weights[name][0].stride()),
                        "ms_per_replay": elapsed,
                        "us_per_layer": elapsed * 1000 / args.layers,
                        "speedup": baseline / elapsed,
                        "samples_ms": times,
                    }
                ),
                flush=True,
            )

        if args.profile_dir:
            for name, graph in graphs.items():
                with torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU,
                    ],
                    record_shapes=True,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        f"{args.profile_dir}/{args.op}/bs{tokens}/{name}"
                    ),
                ):
                    graph.replay()
                    torch.npu.synchronize()


if __name__ == "__main__":
    main()
