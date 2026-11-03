# start

简介：本项目在 5080 测试，cuda12.9。但 agent 阅读采用另外的设备，需要注意。

## 1. cmd
```sh
# 设置 CUDA 环境并检查
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

nvcc --version
cmake3279 --version
g++ --version

# 配置独立构建目录
cmake3279 -S . -B build-sm120-cuda129 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 \
  -DCUTLASS_NVCC_ARCHS=120a \
  -DCUTLASS_ENABLE_EXAMPLES=ON \
  -DCUTLASS_ENABLE_TESTS=OFF

# 只编译第 79a 样例
cmake3279 --build build-sm120-cuda129 \
  --target 79a_blackwell_geforce_nvfp4_bf16_gemm \
  --parallel 4

# 测试
CUDA_VISIBLE_DEVICES=00000000:5E:00.0 \
  ./build-sm120-cuda129/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm \
  --m=2048 \
  --n=2048 \
  --k=2048 \
  --iterations=100

CUDA_VISIBLE_DEVICES=00000000:5E:00.0 \
  ./build-sm120-cuda129/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm \
  --m=8192 \
  --n=8192 \
  --k=8192 \
  --iterations=100
```
## 2. 功能翻译

```sh
\brief 一个使用 CUTLASS 用于 NVIDIA Blackwell SM120 架构的 GEMM 示例。

    本示例展示了在NVIDIA Blackwell SM120 架构上实例化和运行块级 NVFP4 GEMM 的简单方法。
    该内核针对 GeForce RTX 5 系列GPU进行了优化。

    Blackwell SM120 CUTLASS 内核采用了新的块缩放张量核心 MMA 指令（mma.sync.aligned.block_scale）。
    NVFP4 MMA 的吞吐量是 MXFP8 MMA 的 2 倍，而 Ada Tensor Core FP8 MMA 的吞吐量是 4 倍。
    （参见 https://docs.nvidia.com/cuda/parallel-thread-execution）

    该内核利用：
    1. Warp-Specialized 持久内核设计，支持合作式和乒乓式内核调度，Hopper 引入。
    2. 新的 SW controlled 动态调度器基于集群发射控制（见 https://docs.nvidia.com/cuda/parallel-thread-execution）。
    3. 块尺度张量核心MMA指令
    4. 尾声优化

    请注意，GeForce RTX 50系列 GPU 不支持：
    1. TMA负载的多播功能。cluster 形状必须是 1x1x1。
    2. 动态数据类型。

    使用情况：

      $ ./examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm --m=2048 --n=2048 --k=2048

```


## 3. 样例解读

## 3.1 79a

对应源码：[79a_blackwell_geforce_nvfp4_bf16_gemm.cu](../examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)。

这个样例演示在 Blackwell GPU 上执行带分块缩放的 NVFP4 矩阵乘法：A、B 使用 FP4 数据并配合缩放因子 SFA、SFB，内部使用 FP32 累加，C、D 使用 BF16。计算可以理解为 `D = alpha × (缩放后的 A × 缩放后的 B) + beta × C`；默认 `alpha=1`、`beta=0`，即输出矩阵乘法结果。源码通过 `CollectiveMainloop` 配置乘法主循环，通过 `CollectiveEpilogue` 配置结果缩放、加 C 和写回，再组合成可调用的 `Gemm`。

`main()` 负责程序入口与流程控制：检查 CUDA 编译版本和当前 GPU 的计算能力（支持 SM120/SM121），解析命令行参数，按需打印帮助，最后调用 `run<Gemm>(options)` 完成计算、正确性验证和性能测试。

各子函数的职责如下：

| 函数 | 作用 |
| --- | --- |
| `Options::parse()` | 读取 `m/n/k`、`alpha/beta` 和测试迭代次数；默认尺寸为 `1024×1024×1024`，迭代 10 次。 |
| `Options::print_usage()` | 输出参数说明和运行示例。 |
| `Options::gflops()` | 按 `2×M×N×K / 耗时秒数 / 10^9` 计算吞吐量，单位为 GFLOPS。 |
| `initialize_block()` | 根据元素类型选择随机数范围，在主机端填充输入数据或缩放因子。 |
| `initialize()` | 创建矩阵及缩放因子的布局，分配主机和设备内存，调用 `initialize_block()` 填充 A、B、C、SFA、SFB，并将输入复制到 GPU。 |
| `args_from_options()` | 将问题尺寸、设备数据指针、步长、缩放因子布局和 `alpha/beta` 打包为 GEMM 调用参数。 |
| `make_iterator()` | 将原始指针转换为 CuTe 可用的迭代器，供参考计算构造张量，也支持 FP4 这类不足一个字节的元素。 |
| `verify()` | 在 CPU 上用相同输入和缩放因子计算参考结果，将 GPU 输出复制回主机，检查两者逐元素相等且结果范数非零。 |
| `run<Gemm>()` | 串起完整执行过程：初始化数据、组装参数、申请工作区、检查配置是否支持、初始化并运行 GEMM；同步后验证结果，通过后再循环计时，输出平均耗时和 GFLOPS。 |

阅读时可以沿着 `main() → run() → initialize() / args_from_options() → gemm.run() → verify()` 理解调用顺序。首次运行用于预热和正确性验证，不计入后续性能循环；计时循环中每轮都会调用 `gemm.initialize()` 和 `gemm.run()`。前面的 `2048` 与 `8192` 两组命令使用相同流程，只是矩阵尺寸不同。

#### NCU

```sh
#### tensor core 利用率查询

sudo \
CUDA_VISIBLE_DEVICES=00000000:5E:00.0 \
/usr/local/cuda-12.9/bin/ncu \
  --launch-skip 1 \
  --launch-count 1 \
  --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed \
  ./build-sm120-cuda129/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm \
  --m=2048 \
  --n=2048 \
  --k=2048 \
  --iterations=10
  
#### tensor core 利用率查询
sudo \
CUDA_VISIBLE_DEVICES=00000000:5E:00.0 \
/usr/local/cuda-12.9/bin/ncu \
  --set full \
  --launch-skip 1 \
  --launch-count 3 \
  -o docs_step/log/79a_2048_full \
  ./build-sm120-cuda129/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm \
  --m=2048 \
  --n=2048 \
  --k=2048 \
  --iterations=10
```

## 3.2 79b

## 3.3 79b

## 3.4 79b


## 4. 实验分析

以下命令在项目根目录执行，编译并测试 79a～79d 四个样例。

| 样例 | 功能 |
| --- | --- |
| 79a | NVFP4 输入，BF16 输出的 GEMM。 |
| 79b | NVFP4 输入，NVFP4 输出的 GEMM，输出包含分块缩放因子。 |
| 79c | A 使用 MXFP8、B 使用 MXFP6 ，输出为 BF16 的混合精度 GEMM。 |
| 79d | NVFP4 分组 GEMM，一次调度多组矩阵乘法，输出为 NVFP4；程序分别测试 Cooperative 和 Pingpong 两种调度。 |

```sh
# 编译第 79a、79b、79c、79d 样例
cmake3279 --build build-sm120-cuda129 \
  --target \
    79a_blackwell_geforce_nvfp4_bf16_gemm \
    79b_blackwell_geforce_nvfp4_nvfp4_gemm \
    79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm \
    79d_blackwell_geforce_nvfp4_grouped_gemm \
  --parallel 4

# 指定测试 GPU 和可执行文件目录
export CUDA_VISIBLE_DEVICES=00000000:5E:00.0
example_dir=./build-sm120-cuda129/examples/79_blackwell_geforce_gemm

# 四个样例分别测试 2048 和 8192 两种尺寸，每次性能测试迭代 100 次
for size in 2048; do
  # 79a：NVFP4 -> BF16
  "$example_dir/79a_blackwell_geforce_nvfp4_bf16_gemm" \
    --m="$size" --n="$size" --k="$size" \
    --alpha=1 --beta=0 --iterations=100

  # 79b：NVFP4 -> NVFP4
  "$example_dir/79b_blackwell_geforce_nvfp4_nvfp4_gemm" \
    --m="$size" --n="$size" --k="$size" \
    --alpha=1 --beta=0 --iterations=100

  # 79c：MXFP8 × MXFP6 -> BF16
  "$example_dir/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm" \
    --m="$size" --n="$size" --k="$size" \
    --alpha=1 --beta=0 --iterations=100

  # 79d：分组 NVFP4 GEMM，每组均使用当前尺寸
  "$example_dir/79d_blackwell_geforce_nvfp4_grouped_gemm" \
    --m="$size" --n="$size" --k="$size" --groups=2 \
    --alpha=1 --beta=0 --iterations=100
done
```

上述指令均保留默认的正确性校验。79d 显式设置 `--groups=2`，表示执行两组相同尺寸的 GEMM；其计算量和内存占用随组数增加。`--iterations=100` 用于性能计时，若只需验证正确性，可改为 `--iterations=0`。
