# RLC 布局优化增强（metax 后端）文档总览

这套文档面向两类读者：

- **不写编译器的工程师**：想知道「为什么我的 kernel 某次版本变快了/变慢了」「某个开关要不要动」；
- **维护 metax 后端的开发者**：想知道这些检查各自守住了什么、边界在哪、出问题先看哪里。

每篇只讲一件事，尽量不用术语；必须出现的术语在第一次出现时给出中文解释和英文原文。读不下去的时候可以直接跳到你要查的那一篇，各篇之间没有强依赖。

---

## 0. 一分钟版：这套东西在做什么

GPU 上做计算之前，编译器要把每个张量（tensor）切分给成千上万个线程，并且决定「哪个线程算哪几个元素」。这个分配方案叫**布局（layout）**。

同一个算子图，往往可以有好几种合理布局。不同算子又有各自的偏好：

- **load / store**（读写显存）喜欢布局让同一线程拿到的地址是**连成一片**的，这样一次内存事务能搬回一整段，效率高；
- **矩阵乘（dot）** 喜欢 tensor core 需要的分块方式；
- **归约（reduce）**、**广播（broadcast）** 又各有偏好。

当相邻两个算子的偏好不一致时，编译器就得插入一次**布局转换（`convert_layout`）**——把数据按新方案重新分配。这个动作通常要借共享内存（shared memory）中转一趟，是实实在在的开销。

所以 `RemoveLayoutConversions` 这个 pass 的工作就是：**在不破坏访存模式的前提下，消除尽可能多的 `convert_layout`**。

它原本的规则比较朴素（「谁像锚点就听谁的」）。这套 RLC 增强（RLC = Remove Layout Conversions）把它换成一个带代价模型的版本，并在实践中发现了四类会误伤的判断，逐条补了保护：

| 编号 | 功能 | 一句话 | 默认 | 详细文档 |
|---|---|---|---|---|
| 1 | 循环内的占用上限 | 循环里把更「碎」的布局塞给一个 load，会让每个线程每轮多搬一堆数据 | 开 | [01-load-footprint-guard.md](01-load-footprint-guard.md) |
| 2 | 分阶段开关 | 六个开关能分别开/关，且能不改代码用环境变量覆盖 | 开 | [02-phase-switches.md](02-phase-switches.md) |
| 3 | 别把 load 的消费者搬走 | load 自己不能改布局，却把它的下游全改掉，等于白搬一趟 | 开 | [03-stranded-consumers.md](03-stranded-consumers.md) |
| 4 | 决策日志 | `RLC_DEBUG=1/2` 把每个「接受/拒绝」的理由打到 stderr | 关（需设环境变量） | [04-debug-logging.md](04-debug-logging.md) |
| 5 | 真实地址连续性 | 布局上写得下 N 个连续元素，不等于内存里真的连续 | 开 | [05-real-contiguity.md](05-real-contiguity.md) |
| 6 | 没有循环时的裁决 | 成本只按「循环内」计权重，完全展开的 kernel 里比较永远打平 | 开 | [06-conflict-tie-break.md](06-conflict-tie-break.md) |

> 默认值说明：metax 的 `python/triton/knobs.py` 里 `rlc_enhance = env_bool("FLAGTREE_RLC_ENHANCE", True)`，因此**主开关默认打开**，上表前 3 项、第 5、6 项都随之生效；只有调试日志需要你显式设 `RLC_DEBUG`。

---

## 1. 需要的最小背景

如果下面三小节你能读通，六篇文档都能读；读不通也没关系，每篇会就地再解释一次。

### 1.1 布局（layout / encoding）到底规定了什么

一个张量（比如 `64 × 64` 的矩阵）要分给一个 CTA（线程块，比如 4 个 warp × 64 线程 = 256 个线程）。布局要回答四个问题：

- `sizePerThread`：每个线程沿每个维度**连续**拿几个元素；
- `threadsPerWarp`：一个 warp 内线程沿每个维度怎么排；
- `warpsPerCTA`：warp 之间沿每个维度怎么排；
- `order`：维度里哪个是「内存最快变化」的方向（用来判断谁和地址连续性相关）。

在日志里你会看到这种紧凑写法：

```
b[1,4]/t[1,32]/w[4,1]/o[1,0]
```

读作：「每线程连续 `[1,4]` 个元素，warp 内 `[1,32]` 排布，4 个 warp 沿第 0 维，内存最快方向是第 1 维」。`slice(0, ...)` 表示从某个父布局里沿第 0 维切出来的一层（常见于归约结果、标量广播）。

### 1.2 `convert_layout` 为什么贵

当生产者布局和消费者布局不一致，就得插一次 `convert_layout`。它一般要实现成「写共享内存 → 同步 → 按新方案读回来」，所以：

- 在循环里，它**每一轮都要付一次**；
- 它还会带来同步点，打断流水。

因此代价模型里 `convert` 的权重按**循环层数**放大（见 1.3）。

### 1.3 代价模型的口径

这个 pass 的增强版不估算周期数，只数「`convert_layout` 的加权个数」：

- 每个 `convert` 有一个单位成本 `getConvertUnitCost`，与元素字节数相关（`32 × 字节数`，最小为 1）；
- 位置在循环里就乘权重：`getScopeLoopFrequencyWeight` 返回 `8^depth`（最多算 3 层循环），不在循环里返回 `1`；
- 提案被接受的条件是「改完之后的总成本**严格变小**」。

这套口径简单、可预测，但有两个天生的盲点，正好是本套文档第 5、6 篇要补的：

- 它只看得懂「`convert` 变少」，看不懂「访存变差」——于是第 5 篇补了真实地址检查；
- 它的权重只有「在循环里/不在循环里」两档——完全展开的 kernel 里所有位置权重都是 1，比较会退化成打平，于是第 6 篇补了平局裁决。

---

## 2. 这套增强管的是哪些阶段

`RemoveLayoutConversions` pass 跑在 metax pipeline 的多个位置（`third_party/metax/backend/compiler.py` 的 `make_ttgir` 里会调用多次）。每次运行内部有四个阶段：

1. **前向传播（Phase 1）**：从「锚点算子」（load/store/dot 等布局不可动的）出发，把布局顺流推到下游，为每个值收集**候选布局集合**；
2. **反向传播（Phase 1b，`propagateLayoutBackward`）**：从写着 convert 的尾巴往回推，试图让上游直接产出目标布局；
3. **小分量求解（Phase 2，`solveSmallComponents`）**：针对「一小撮互相连接的值」，尝试把它们统一到某个布局，从而一次消掉几个 convert，然后**用代价模型验收**；
4. **冲突消解（`resolveConflicts` / `resolveConflictsByComponent`）**：一个值若有多个候选布局，最后要挑一个。

其后还有两轮 remat 阶段（反向重物化、写回布局重物化），以及收尾的 hoist/cleanup。

六个功能分布在这些阶段里：

```
Phase 1/1b/2  ── 功能 1（占用上限）、功能 3（别搬走消费者）、功能 5（真实连续性）
冲突消解      ── 功能 6（平局裁决）
remat 阶段    ── 功能 1 的第二处、功能 5 的第二处
全过程        ── 功能 2（开关）、功能 4（日志）
```

---

## 3. 开关与观测速查

### 3.1 环境变量

| 变量 | 作用 | 取值 | 默认 |
|---|---|---|---|
| `FLAGTREE_RLC_ENHANCE` | 主开关 | `on/true/1` 或 `off/false/0` | 开 |
| `FLAGTREE_RLC_ENHANCE_COST_BASED` | 代价模型消解冲突 | 同上 | 开 |
| `FLAGTREE_RLC_ENHANCE_BACKWARD_PROPAGATION` | 反向传播阶段 | 同上 | 开 |
| `FLAGTREE_RLC_ENHANCE_SMALL_COMPONENT_SOLVING` | 小分量求解阶段 | 同上 | 开 |
| `FLAGTREE_RLC_ENHANCE_STORE_LAYOUT_REMAT` | 写回布局重物化（也会打开 duplicable-load 重物化） | 同上 | 开 |
| `FLAGTREE_RLC_ENHANCE_MAX_THREAD_REACH` | 第 5 篇的「单个线程一次访问最远跨多少字节」阈值 | 正整数（字节） | `49152`（48 KiB） |
| `RLC_DEBUG` | 决策日志级别 | `1` 决策 / `2` 逐值细节 | 关（0） |

优先级：**环境变量 > pass 选项**；环境变量写了无法识别的值（例如 `yes`）会当作没写。
细节与坑（哪些变量是进程内只读一次的）请看 [02-phase-switches.md](02-phase-switches.md)。

### 3.2 打开日志

```bash
RLC_DEBUG=1 python my_kernel.py 2>&1 | grep '\[rlc\]'
RLC_DEBUG=2 python my_kernel.py 2>&1 | grep -E '\[rlc|\[rlc-remat-load\]'
```

日志标签速查见 [04-debug-logging.md](04-debug-logging.md)。

### 3.3 怀疑某个功能误伤时

先用对应阶段的环境变量把它单独关掉，做 A/B：

```bash
# 怀疑第 5 篇的真实连续性检查过于保守（保留了不该保留的 convert）
FLAGTREE_RLC_ENHANCE_MAX_THREAD_REACH=2147483647 python bench.py

# 怀疑第 3 篇的「别搬走消费者」否决掉了本该做的优化
FLAGTREE_RLC_ENHANCE_SMALL_COMPONENT_SOLVING=0 python bench.py
```

关掉某一阶段后行为会退回到「没有该阶段」的版本，适合快速定位，不适合作为长期配置。

---

## 4. 代码在哪儿

全部逻辑集中在同一个文件（metax 后端是 spec_cpp 的私有副本）：

```
third_party/metax/spec_cpp/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp
```

共享的环境变量登记表：

```
include/triton/Tools/Sys/GetEnv.hpp
```

Python 侧的开关入口：

```
python/triton/knobs.py            # metax_knobs.rlc_enhance
third_party/metax/python/src/passes.cc         # add_remove_layout_conversions 绑定
third_party/metax/backend/compiler.py          # 在 make_ttgir 中多次加入该 pass
```

> 注意：`lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp`（主树）是 PPU 分支在用的版本，metax 构建使用的是 `third_party/metax/spec_cpp/...` 这份副本。**改这里不会影响 metax；改那边也不会。** 两边逻辑目前一一对应。

---

## 5. 与 rlc_ppu 分支的对应关系

这套功能最初在 `rlc_ppu` 分支上开发，随后移植到 metax（去掉 `__PPU__` 宏、对 metax 无条件生效）：

| 文档 | rlc_ppu commit | 移植差异 |
|---|---|---|
| 01 | `d61e4a4c9` | 去掉 `#ifdef __PPU__`，补 `llvm/Support/MathExtras.h` |
| 02 | `6473c6721` | 无（纯共享代码） |
| 03 | `f0513d863` | 去掉 `#ifdef __PPU__` |
| 04 | `3d52ca1cf` | 去掉 `#ifdef __PPU__` |
| 05 | `65eb83320` | 去掉 `#ifdef __PPU__`；注释里的 PPU 案例标为「移植来源」，阈值仍为 48 KiB 但可用环境变量重标定 |
| 06 | `059d3f2d2` | 去掉 `#ifdef __PPU__` |

移植时的 metax 适配要点（重要，且容易忘）：

- **warp 宽度是 64 不是 32**：所有「每个线程均摊多少元素」的计算依赖 `ttg.threads-per-warp`，metax 的 `make_ttgir` 用 64 传给 `add_convert_to_ttgpuir`，因此算出来的是真实线程数；
- **48 KiB 阈值来自 PPU 的实测标定**，metax 沿用同一默认值，需要时用 `FLAGTREE_RLC_ENHANCE_MAX_THREAD_REACH` 重新标定，不必改代码。

---

## 6. 已知的共同边界

写在这里避免每篇重复：

1. **这些检查都是「保守的方向」**：拿不准时倾向于**保留 `convert`**。所以它们的天花板是「少做错事」，不是「多做优化」。想看性能上限，应该去改代价模型本身，而不是关掉这些保护。
2. **代价模型只看 `convert` 个数与循环权重**，不建模寄存器压力、bank conflict、流水深度。所有判断都是这个口径下的局部最优。
3. **它们只影响「布局怎么选」，不改变语义**：任何被拒绝的提案，只是让某个值保持原来的布局，不会产生错误结果。真出问题时的表现是「性能不如预期」，不是「算错」。
4. **阈值是经验值**：`×2`（第 1 篇）、`48 KiB`（第 5 篇）、`4 跳`（第 3 篇）都来自实测，换硬件、换算法族可能不再最优，因此都做成了可调或易改的形式。
5. **本仓库当前没有针对这些检查的单元测试**，验证方式是端到端 kernel 的 A/B 性能对比 + `RLC_DEBUG` 日志比对。改这些代码时请自己准备最小复现 kernel。
