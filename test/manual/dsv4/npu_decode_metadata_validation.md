# NPU DSV4 decode compression metadata fusion

普通 decode 的 NPUGraph replay 每步都会调用
`DeepseekV4AscendAttnBackend._refresh_graph_decode_compress_1d_direct`。
本次将其中的 C4/C128 loc 复制和清尾、压缩边界筛选、位置稳定压紧、
`start_pos` / `seqused` 更新合成一次 Triton kernel 调用，直接写入已有图输入。
prefill、target verify 和 C128 sidecar/refcount 更新不属于本次范围。

相比 C128 边界批处理，这一改动影响持续 decode 的每一步，因此优先验证它对
TPOT 的作用。已有重型 profile 中该主机区间约 12–16 ms；这是选点依据，
不能当成无 profiler 的耗时，也不能当成预计端到端收益。

## 异步依赖

```text
主机：原位置调用 metadata refresh → 提交后续 metadata → 提交 graph replay
                       │                    │                   │
当前 NPU stream：读取 seq/loc → 融合 kernel → 后续 metadata → 图内消费者
                                    │
                                    └→ 原有固定地址的六个输出 buffer
```

融合调用异步下发，返回不代表设备已完成。同 stream 顺序保证后续消费；
调用方已有的跨 stream 依赖仍需保留。热路径不新增 `synchronize()`、
`.item()`、D2H、临时 tensor 或私有 stream。保留图外原调用位置，
以维持当前 CPU/NPU 重叠关系。

每个 decode graph bucket 在捕获前预热。loc 数量是运行时参数，
压缩边界引起的数量变化不会单独产生 JIT 变体；dtype、stride 或 shape
变化仍可能需要新编译，测试前应充分预热真实 workload。

## 正确性与端到端 A/B

在具备项目依赖、Triton Ascend 和 NPU 的环境运行：

```bash
python test/registered/unit/npu/attention/test_npu_ascend_dsv4_backend.py TestNPUDecodeCompressionMetadata -v
```

测试覆盖缺失 loc、padding/idle、不同压缩比例、非连续输入、整数 dtype、
不同 batch 大小，以及非默认 stream 上 metadata 更新后 graph replay
读取新数据、旧尾部清零和输出地址不变。本机没有 NPU，尚未验证实际编译、
设备执行或端到端加速。

使用同一代码，分别在以下环境变量下重启服务，再运行相同 benchmark：

```bash
export SGLANG_NPU_DSV4_FUSED_DECODE_METADATA=0  # 原路径
export SGLANG_NPU_DSV4_FUSED_DECODE_METADATA=1  # 融合路径，默认开启
```

两行分别用于两次服务启动，不要在同次启动前连续执行。保持模型、TP、
请求输入、并发、输出长度、graph bucket 和其他优化开关一致。

1. 先做输出正确性检查，并预热相同 batch 和压缩边界。
2. 关闭 profiler，交替运行多轮 A/B，比较客户端 TPOT、请求总耗时和吞吐。
3. 需要解释收益时再采所有 rank 的 profile：比较 metadata 主机提交、
   设备 graph 完成间隔，以及最慢 rank 到达 collective 的时刻。

只有主机区间缩短而设备完成和客户端指标不变，说明收益被其他依赖覆盖，
不能认定端到端优化成功。也不要在每个 metadata 阶段插同步来计时，
否则会破坏原来的重叠关系。
