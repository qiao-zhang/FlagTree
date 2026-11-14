# 功能 4：决策日志（`RLC_DEBUG`）

> 对应代码：`rlcDbgLevel()`、`rlcDbgEnc()` 及 14 处 `[rlc]` 打点、4 处 `[rlc-remat-load]` 打点
> 对应 rlc_ppu commit：`3d52ca1cf`
> 默认状态：**关闭**（未设 `RLC_DEBUG` 时零输出）

---

## 一句话

给这套布局优化加一套**最轻量的 stderr 打点**：`RLC_DEBUG=1` 看每个决策的结论与理由，`RLC_DEBUG=2` 再看每个值的候选布局与成本明细。不引入任何 IR 属性、不新增 pass 选项，输出就是能 `grep` 的纯文本。

---

## 1. 为什么需要它

### 1.1 这套优化的排查难点

布局优化失败时的表现只有一种：**没有变快**。而「没有变快」的原因可能有一屋子的嫌疑人：

- 提案压根没被生成（走错分支）；
- 提案生成了，但消掉的 convert 数为 0；
- 消掉了 1 个，但被「太弱」规则拦下；
- 被第 3 篇的「上游是个昂贵 load」拦下；
- 被第 5 篇的「真实地址不支持」拦下；
- 过了所有护栏，但代价模型认为「改完更贵」。

这些分支在原始实现里**全部静默**。加上 `LLVM_DEBUG` 形式的 `LDBG` 还不够好用：它要求构建时开 debug、运行时设 `TRITON_LLVM_DEBUG_ONLY`，而 metax 的日常构建并不是这么来的。

### 1.2 为什么不用 MLIR 的 trace 框架

Triton 里有一套 trace 机制（往 IR 上挂属性和 phase 记录）。它很适合「事后分析一份编译产物」，但用在这里有三个不合算：

- 会往 IR 上加属性 → 影响 IR 本身、可能干扰其它 pass 的模式匹配；
- 需要额外的 pass 选项来开启，等于给用户又加一个开关；
- 体积重，看一次日志还得配套工具去解析。

这里要的只是「**跑一遍，用 grep 捞出来看**」。所以最终选择是：直接 `llvm::errs()`，由一个环境变量控制级别。

---

## 2. 两个级别

| 级别 | 你能看到 | 适合 |
|---|---|---|
| `RLC_DEBUG=1` | 决策层：提案规模、消掉几个 convert、接受/拒绝及理由、成本前后对比、冲突消解的总体规模、各阶段入口的提案数 | 判断「为什么没优化」 |
| `RLC_DEBUG=2` | 值层：提案里每个值的 `值 -> 布局`、每个 convert 的计数明细、每个值的候选布局与各自成本、最终选中与当前布局的对比 | 判断「为什么选了这个布局」 |

```bash
RLC_DEBUG=1 python my_kernel.py 2>&1 | grep '\[rlc\]'
RLC_DEBUG=2 python my_kernel.py 2>&1 | grep -E '\[rlc|\[rlc-remat-load\]'
```

级别是「至少」语义：设 `2` 也会输出级别 `1` 的所有内容。

---

## 3. 具体做法

### 3.1 级别读取：进程内只读一次

```cpp
static int rlcDbgLevel() {
  static const int level = [] {
    const char *env = std::getenv("RLC_DEBUG");
    return env ? std::atoi(env) : 0;
  }();
  return level;
}
```

- `static const` + lambda：第一次调用时读环境变量，之后一直用这个值；
- 用 `std::atoi`，所以 `RLC_DEBUG=2`、`RLC_DEBUG=1abc` 都能识别（后者按 1）。设 `0` 或未设 = 关闭；
- **同一进程内改环境变量无效**，必须新进程。这是刻意换取「零成本」：正常运行路径上只有一次初始化 + 一次整数比较。

### 3.2 布局的紧凑打印

原始布局打印出来长这样：

```
#ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
```

一行日志里出现几次就没法看了。所以有一个 `rlcDbgEnc`：

| 情况 | 输出 |
|---|---|
| blocked 布局 | `b[1,4]/t[1,32]/w[4,1]/o[1,0]` |
| slice 布局 | `slice(d0,b[1,4]/t[1,32]/w[4,1]/o[1,0])`（递归打印父布局） |
| 空 | `<null>` |
| 其它（MMA、linear 等） | 调用 `print` 打印完整属性 |

读法（务必先看 README 的 1.1 节）：`b` = sizePerThread，`t` = threadsPerWarp，`w` = warpsPerCTA，`o` = order。`o[1,0]` 表示第 1 维是内存最快变化的方向。

### 3.3 打点清单

**级别 1（决策）与级别 2（值细节）共 14 处 `[rlc]`：**

| 日志前缀 | 位置 | 说明 |
|---|---|---|
| `[rlc] cost before=… after=… accept=…` | `shouldAcceptProposal` | 代价模型的结论：改前/改后加权 convert 成本 |
| `[rlc] proposal n=…` | `commitIfProfitable` | 一个合并后提案包含多少个值 |
| `[rlc] proposal removed=…` | 同上 | 这个提案状态消掉几个 convert |
| `[rlc] reject no-removed-convert` | 同上 | 一个都消不掉 → 否决 |
| `[rlc] reject untaggable-load-boundary: …` | 同上 | 第 3 篇的否决（上游是动不了的昂贵 load） |
| `[rlc] accept n=… removed=…` | 同上 | 提案通过全部护栏，即将落地 |
| `[rlc] entry=reduceTail n=…` | `solveSmallComponents` | 归约写回尾阶段收集到多少提案 |
| `[rlc] entry=loopTail n=…` | 同上 | 循环写回尾阶段 |
| `[rlc] entry=storeTail n=…` | 同上 | 直接 store 尾阶段 |
| `[rlc] entry=producerSide n=…` | 同上 | 生产者侧链阶段 |
| `[rlc] rc layouts=… multi-candidate=…` | `resolveConflicts` | 冲突消解：多少值有多个候选布局 |
| `[rlc] rc <值> minCost=… candidates=…` | 同上（级别 2） | 单个值的最低成本与候选数 |
| `[rlc] rc <值> chosen=… current=… tied=… CHANGED/same` | 同上（级别 2） | 最终选中哪个、是否与当前不同 |
| `[rlc] cvt src=… dst=… propSrc=… propDst=… counted=…` | `countConvertsRemovedByProposal`（级别 2） | 逐个 convert 记账：是否计入「消掉」 |

**4 处 `[rlc-remat-load]`：**

| 日志 | 位置 | 输出方式 |
|---|---|---|
| `[rlc-remat-load] enter ptr=… cur=… tgt=…` | `isDuplicableLoadInEncoding` 入口 | `errs`（级别 2） |
| `[rlc-remat-load] reject widen-past-real-contiguity: …` | 同上，第 5 篇的否决 | `errs`（级别 1） |
| `[rlc-remat-load] reject: increased footprint above ideal * 2` | 同上，第 1 篇的否决 | **`LDBG`**（需 `LLVM_DEBUG` 构建） |
| `[rlc-remat-load] reject: footprint exceeds 256 bits` | 同上 | **`LDBG`** |

> 注意后两条：它们是 `LDBG`，普通构建里**不输出**。这是当前实现的疏漏——同一类判断用了两套输出方式。若要统一，把 `LDBG(...)` 换成 `if (rlcDbgLevel() >= 1) llvm::errs() << ...` 即可。

另外还有 **12 处 `[rlc]` 来自第 5 篇**，分三类：

- **6 处**是 `real-contiguity` 判定过程本身：`skip(原因)` / `skip(no-widen)` / `reject(no-axisinfo)` / `allow(coalesced)` / `reject(no-pitch)` / 汇总行；
- **1 处**是闭包路径上的否决消息 `reject widen-past-real-contiguity: …`；
- **5 处**在 remat/写回阶段：`backRemat try`、`backRemat removed|kept`、`writeback skip(non-mma)`、`backRemat duplicable=`、`storeLayoutRemat changed=`。

它们的含义见第 5 篇文档。整个文件里 `[rlc]` 合计 26 处（14 + 12），`[rlc-remat-load]` 合计 4 处。

---

## 4. 典型排障流程

### 4.1 场景：「某 kernel 变慢了，想知道优化器做了什么」

```bash
RLC_DEBUG=1 python bench.py 2>&1 | grep '\[rlc\]'
```

你会拿到一串形如：

```
[rlc] rc layouts=42 multi-candidate=7
[rlc] entry=reduceTail n=1
[rlc] proposal n=2
[rlc] proposal removed=1
[rlc] cost before=64 after=32 accept=1
[rlc] accept n=2 removed=1
```

读法：有 42 个值被跟踪、其中 7 个有多个候选；归约尾阶段收集到 1 个提案；该提案含 2 个值、能消 1 个 convert；成本从 64 降到 32；接受。

如果看到的全是 `accept=0` 或 `reject ...`，问题就在「没优化成」，往下一节走。

### 4.2 场景：「提案被否了，是谁否的」

按日志出现的顺序对号入座：

| 最后一行是 | 谁否的 | 下一步 |
|---|---|---|
| `reject no-removed-convert` | 提案状态上没消掉任何 convert | 说明这组值的布局关系没被提案拉平，属正常；看级别 2 的 `[rlc] cvt ... counted=false` 找原因 |
| `reject untaggable-load-boundary: …` | 第 3 篇 | 用 `03-stranded-consumers.md` 的判断表核对；必要时临时关该阶段做对照 |
| `cost before=A after=B accept=0` | 代价模型 | 说明改完更贵或打平。注意打平本身也是「不接受」，平局场景见第 6 篇 |
| 什么都没有（连 `proposal n=` 都没有） | 提案没被生成 | 说明模式匹配阶段就没命中，与本套功能无关，去看 `solveSmallComponents` 的收集条件 |

第 5 篇的否决走另一组标签：`[rlc] reject widen-past-real-contiguity: …`（闭包路径）和 `[rlc-remat-load] reject widen-past-real-contiguity: …`（remat 路径）。

### 4.3 场景：「选了奇怪布局」

```bash
RLC_DEBUG=2 python bench.py 2>&1 | grep -A3 'rc .*minCost'
```

会看到某个值的候选集与各自成本：

```
[rlc] rc %12 = arith.addf ... minCost=32 candidates=2
        b[1,4]/t[1,32]/w[4,1]/o[1,0] cost=32
        b[1,1]/t[1,32]/w[4,1]/o[1,0] cost=64
[rlc] rc %12 = arith.addf ... chosen=b[1,4]/... current=b[1,1]/... tied=1 CHANGED
```

意义很直接：便宜的候选被选中、且确实变了布局。如果 `CHANGED` 的那一行让你意外，看这两个候选的 `cost` 差异——代价模型就是这么算的（每 convert 单位成本 × 循环权重）。

---

## 5. 局限与边界

1. **级别与阈值都在进程内缓存**：`RLC_DEBUG` 只读一次，中途改无效；同理第 5 篇的 `FLAGTREE_RLC_ENHANCE_MAX_THREAD_REACH`。
2. **输出无加锁、无标识**：打点直接用 `llvm::errs()`，没有 mutex。并行编译多个 kernel（多进程）或 pass 在多线程环境跑时，行会交错；日志里也没有 kernel 名/时间戳。量产环境里建议配合单 kernel 复现脚本，或把 stderr 按进程拆开。
3. **没有「按值过滤」能力**：无法只打印某个特定的值/算子，级别 2 在真实 kernel 上会非常长。常见做法是先 `grep '[rlc] cost'` 看结论，再 `grep -A` 展开上下文。
4. **两类判断用了两套输出方式**（`errs` 与 `LDBG`），见 3.3 的提醒。
5. **不覆盖非 RLC 路径**：原始（未增强）实现的决策依然静默；本日志只描述增强逻辑。
6. **不含耗时信息**：不知道每个阶段花多久。需要计时请用 `MLIR_ENABLE_TIMING` 或 `LLVM_ENABLE_TIMING`。
7. **对性能的影响可忽略但不为零**：未开启时是「一次静态初始化 + 一次整数比较」；级别 2 在大型 kernel 上会产生大量 I/O，实测编译时间会明显变长——不要在生产编译脚本里常开。
8. **日志文本不是稳定接口**：字段和措辞可能随实现改动，不要写脚本严格解析（除了 `grep` 关键字）。

---

## 6. 代码导读

1. `rlcDbgLevel()` / `rlcDbgEnc()`（文件顶部匿名命名空间，`#ifdef __FLAGTREE_RLC_ENHANCE__` 之外，所以始终可用）；
2. `shouldAcceptProposal` 结尾——最小的一个打点，适合作为模板；
3. `commitIfProfitable` ——打点最密集的地方，也是读懂整套流程的最佳入口：提案规模 → 消掉几个 → 各护栏 → 成本 → 接受/否决；
4. `resolveConflicts` ——看「一个值有多个候选布局时怎么选」；
5. `solveSmallComponents` 的四个 `entry=...` ——看各收集阶段捞到多少提案。

移植说明：metax 副本删掉了 `#ifdef __PPU__`（原文有些打点包在 PPU 条件里），使日志对 metax 始终可用；同时补了 `#include <cstdlib>`（`std::getenv` / `std::atoi`）。
