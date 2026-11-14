# 功能 1：循环里别把「更碎」的布局塞给 load（占用上限）

> 对应代码：`collectProducerClosure`、`isDuplicableLoadInEncoding`、`isBackwardPropagationSeedCandidate`
> 对应 rlc_ppu commit：`d61e4a4c9`
> 默认状态：**开启**，无独立开关（受 `FLAGTREE_RLC_ENHANCE` 主开关及其阶段开关约束）

---

## 一句话

在循环里，把「每个线程负责更多元素」的布局强加给一个访存 load，会让这个 load 每轮多搬一堆数据；省下的那个 `convert_layout` 不值这个价，所以**直接拒绝**这种提案。

---

## 1. 问题长什么样

### 1.1 一个具体的场景

假设有一段循环里的 kernel 片段（简化）：

```python
for i in range(0, N, BLOCK):
    x = tl.load(ptr + i + tl.arange(0, BLOCK))   # 连续访存，布局喜欢「每线程少而连」
    y = x * 2.0                                   # 纯逐元素
    z = tl.sum(y[None, :] * w, axis=1)            # 归约/矩阵乘，布局喜欢别的切法
    out[i] = z
```

`tl.load` 那一行是最值钱的地方：它的布局要让每个线程握着的地址尽量连成一片，这样一次内存事务能搬一整段。

而归约、矩阵乘那边的算子，希望的是另一种切法。两者不一致时，编译器插一个 `convert_layout` 来回倒一次数据——**这是正常操作**，也是这个 pass 想消掉的目标。

问题出在「怎么消」。一种看起来诱人的方案是：**别倒数据了，让 `tl.load` 直接产出下游想要的布局**。这样 convert 就没了。

如果下游布局是「每个线程只拿 1 个元素、线程之间交错」，那没事——load 只是换了个取数顺序。但如果下游布局是「每个线程连续拿 4 个、8 个甚至更多元素」（比如 `sizePerThread=[1,8]`），代价就变了：

- load 现在**每个线程要搬 8 个元素**，寄存器占用上升，每次内存请求覆盖的跨度变大；
- 这个 load 在循环里，**每一轮都付一次**；
- 省下的只是**一个** `convert_layout`。

一比就亏了。

### 1.2 为什么原来的检查挡不住

这个 pass 在决定「能不能改一个 load 的布局」时，原有的护栏是：

- `preservesWritebackMemoryAccess`：新布局不能**丢掉**访存连续性（只防「变差」，不防「每个线程拿太多」）；
- `widensPastVectorAccess`：新布局不能让向量宽度无节制变大（它有 128 位这条线，但对「连续但每线程管很多元素」的情况并不敏感，第 5 篇会细讲）；
- remat 路径上还有一条绝对上限：每个线程的占用不超过 256 位（32 字节）。

这些护栏都在问「访存模式有没有变坏」，没人问「**每个线程分摊到的工作量是不是离谱地超出平均**」。本功能就是补上这最后一问。

---

## 2. 判断方法：跟「理想均摊」比

### 2.1 理想均摊是什么

如果一个张量有 `numElements` 个元素、可用线程有 `numThreads` 个，那么**完全平均**地分，每个线程应该拿：

```
idealPerThread = ceil(numElements / numThreads)
```

它是一个很容易算、也不偏袒任何布局的基准线：**没有哪个线程本该承担超过两倍的平均份额**。

`numThreads` 在代码里这样算：

```cpp
int64_t numThreads =
    lookupNumWarps(defOp) * TritonGPUDialect::getThreadsPerWarp(mod);
```

- `numWarps` 从这个 kernel 的属性里取；
- `threadsPerWarp` 从模块属性 `ttg.threads-per-warp` 取。**metax 上这个值是 64**（`third_party/metax/backend/compiler.py` 里 `add_convert_to_ttgpuir(pm, ..., 64, ...)`），不是 NVIDIA 习惯的 32。所以这里算出来的就是真实的线程数，不会少算一半。

### 2.2 拒绝条件

```
newPerThread > basePerThread          且        newPerThread > 2 × idealPerThread
```

两个条件必须同时成立：

- `newPerThread > basePerThread`：**是这次提案把占用抬高了**。如果新布局反而更省，当然放行。
- `newPerThread > 2 × idealPerThread`：抬到了**平均份额的两倍以上**。`2` 是经验值：一倍以内属于「稍微不均」，实际硬件能吸收；超过两倍就开始明显浪费。

只要两个都成立，就返回 `false`，即「这个提案不成立」。等价于：**宁可留着那个 convert**。

---

## 3. 三个落点（为什么是三处）

同一个判断被放在三个不同的决策口上，因为它们各自都可能做「改 load 布局」这种决定：

### 落点 A：`collectProducerClosure`（前向收集 + 小分量求解）

当「小分量求解」想消掉一串 convert 时，它会先顺着数据流往回收集这一串里所有的生产算子（叫「生产者闭包」）。走到一个昂贵 load 时，如果新布局会把它压爆，就直接判定闭包收集失败，整条提案作废。

这一段**没有日志**（静默拒绝），排查时只能从「提案没被接受」间接看出来。

### 落点 B：`isDuplicableLoadInEncoding`（remat 阶段：复制小 load）

remat 阶段的思路是：与其在一个值上做 convert，不如把上游那个**小** load 复制一份、直接用目标布局 load 出来，两份各喂一边。这只在 load 很便宜时才划算，所以这里本来就有两道门：

- 每线程理想均摊 × 位宽 > 128 位 → 拒绝；
- 循环内，每线程实际占用 × 位宽 > 256 位 → 拒绝（**绝对上限，保留**）。

本功能在中间**插入了同样的 ×2 判断**，并给两条拒绝都加了日志：

```
[rlc-remat-load] reject: increased footprint above ideal * 2
[rlc-remat-load] reject: footprint exceeds 256 bits
```

它们通过 `LDBG` 宏输出，也就是**只有在 `LLVM_DEBUG` 构建开启相应的 debug 类型时才能看到**，普通构建里是空语句。

### 落点 C：`isBackwardPropagationSeedCandidate`（反向传播的播种口）

反向传播会挑一些「写着 convert 的尾巴」当作起点，往回把目标布局推到上游。如果这个 convert 的目标布局在**内存序方向上比源布局更窄**，那就意味着推上去会让上游的访存更碎——不予播种。

这里用的判据是 `getContigAlongMemoryOrder`（沿内存最快维，一个线程手里真正连续的元素个数）：

```cpp
if (getContigAlongMemoryOrder(dstType) < getContigAlongMemoryOrder(srcType))
  return false;
```

它和第 5 篇检查的是同一类风险，但角度不同：这里是「一开始就别往这个方向推」，第 5 篇是「真要改布局了，看看地址支不支持」。

---

## 4. 走查一个例子

设：一个 1 维张量，4096 个元素，kernel 用 4 个 warp，metax 每 warp 64 线程。

```
numThreads      = 4 × 64 = 256
idealPerThread  = ceil(4096 / 256) = 16
```

| 当前布局（basePerThread） | 提案布局（newPerThread） | 判定 | 理由 |
|---|---|---|---|
| 1 | 1 | 放行 | 没有抬高占用 |
| 4 | 2 | 放行 | 反而更省 |
| 4 | 20 | 放行 | 20 > 4，但 20 ≤ 2×16=32，属「稍微不均」 |
| 4 | 40 | **拒绝** | 40 > 4 且 40 > 32，占用是平均份额的 2.5 倍 |

注意第 3 行：**它不是「越接近平均越好」**。功能只在「明显过量」时出手；20 个/线程虽然高于平均 16，但仍被接受，因为宁可让代价模型保留一点调整空间。

---

## 5. 什么时候生效，什么时候不生效

| 条件 | 是否检查 |
|---|---|
| load 在 `scf.for` / `scf.while` 循环内 | **检查** |
| load 在循环外（只执行一次） | **不检查**（一次性开销，放大也认了） |
| load 是「昂贵」访存（`isExpensiveLoadOrStore`） | **检查** |
| load 只有一个元素 / 指针元素数少于线程数（命中率高，便宜） | **不检查** |
| 提案让占用变小或不变 | **不检查**（放行） |
| 目标布局让内存序连续数变窄（落点 C） | **拒绝** |

`isExpensiveLoadOrStore` 的判据（`lib/Dialect/TritonGPU/Transforms/Utility.cpp`）：

1. 指针是 tensor-of-pointer 形式 → 昂贵（直接返回真）；
2. 取的是单个值 → 不昂贵（所有线程读同一个地址，命中缓存）；
3. 指针元素数少于 `numWarps × threadsPerWarp` → 不昂贵（线程比元素多，命中率高）；
4. 其余 → 昂贵。

也就是说，这条护栏主要保护的是「**循环内、真在批量搬数据**」的那类 load，正是最值钱也最容易被误伤的那类。

---

## 6. 局限与边界（要老实说清的部分）

1. **×2 是经验阈值，不是推导出来的**。它在 ppu 的实测 kernel 上表现良好；换到访存带宽/延迟特性差别很大的硬件，最优倍数可能不是 2。想调就得改代码（没做成环境变量）。
2. **`idealPerThread` 假设均匀可切**。如果张量形状很小、或某一维远小于线程数，理想份额本身就会偏离实际，「两倍」的判定会偏松或偏紧。
3. **落点 B 的绝对上限（256 位）与新检查是并列关系**，不是替代：即使 `×2` 判断放行了，只要超过 256 位仍然拒绝。改动这两处时不要只改一个。
4. **落点 A 没有任何日志**。这是当前实现的不足：`RLC_DEBUG` 看不到它的拒绝，调试时只能靠「提案被否 + 排除法」。若要补，位置在 `collectProducerClosure` 的 load 分支里。
5. **不看真实地址**。它只知道「布局要求每线程 N 个连续元素」，不知道这 N 个地址在内存里是否真连续——那是第 5 篇的职责。两者叠加才是完整判断。
6. **只管「抬高占用」这一个方向**。其它形式的恶化（比如线程间分布变得极不均、warp 内地址发散）由 `preservesWritebackMemoryAccess`、`widensPastVectorAccess`、`isOrderIndifferentAccess` 负责，不在此列。
7. **不改变正确性**。被拒绝 = 某个值保持原布局，IR 语义完全不变；最坏情况是「少消了一个 convert」。

---

## 7. 怎么确认它生效了

| 想确认的事 | 手段 |
|---|---|
| 落点 B 拒绝过 | `RLC_DEBUG` 无关；需要 `LLVM_DEBUG` 构建并打开对应 debug 类型，观察 `[rlc-remat-load] reject: increased footprint above ideal * 2` |
| 落点 A 拒绝过 | 目前无法直接观测；用 `RLC_DEBUG=1` 看小分量提案是否被接受，再结合日志里该提案的 encoding 变化反推 |
| 整体行为是否变了 | 对比开关 `FLAGTREE_RLC_ENHANCE_SMALL_COMPONENT_SOLVING=0/1` 生成的 TTGIR（`TRITON_KERNEL_DUMP=1`）与运行时 |

一个实用技巧：`RLC_DEBUG=2` 会打印每个提案把哪个值改成什么布局（`[rlc] proposal n=... ` 后面逐条 `value -> encoding`）。如果某个 load 出现在提案里、且目标 encoding 的 `b[...]`（sizePerThread）明显变大，但最终没被接受，十有八九就是这里的某个落点拦下的。

---

## 8. 代码导读

建议按这个顺序读：

1. `isExpensiveLoadOrStore`（`lib/Dialect/TritonGPU/Transforms/Utility.cpp`）——先搞清「昂贵 load」的定义；
2. `collectProducerClosure` 的 `LoadOp` 分支——落点 A，注意上面还有 `preservesWritebackMemoryAccess` 与 `widensPastVectorAccess` 两道既有检查；
3. `isDuplicableLoadInEncoding`——落点 B，注意它前面还有 128 位理想均摊门、后面还有 256 位绝对门；
4. `isBackwardPropagationSeedCandidate`——落点 C，只有 5 行，最短。

移植说明：metax 副本里删掉了 `#ifdef __PPU__`，把这段判断变成**无条件生效**；另外补了 `#include "llvm/Support/MathExtras.h"`，因为要用 `llvm::divideCeil` 做向上取整除法。其余与 `rlc_ppu` 分支逐字一致。
