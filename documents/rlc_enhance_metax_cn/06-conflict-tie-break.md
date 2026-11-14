# 功能 6：没有循环时的平局裁决

> 对应代码：`LayoutPropagation::resolveConflictsByComponent` 里的 `better` 判定
> 对应 rlc_ppu commit：`059d3f2d2`
> 默认状态：**开启**，受 `FLAGTREE_RLC_ENHANCE_COST_BASED` 控制（代价模型关闭时此路径不运行）

---

## 一句话

代价模型只给「循环内」的 convert 记权重。一个**完全展开（没有循环）**的 kernel 里，所有候选的循环内成本都是 0，原有的比较规则永远打平、永远不做选择；这里补上「平局时也比总成本」的兜底规则。

---

## 1. 问题长什么样

### 1.1 冲突消解在做的事

到编译的最后阶段，很多值身上还挂着**多个候选布局**（前向传播、反向传播、小分量求解各自提过意见）。`resolveConflicts` 要替每个值选一个。

它先把「互相连通、且共享同一套候选布局」的值分成一组（叫**组件**，component）。连通的意思是：循环携带值、if 分支的两侧、逐元素算子链……这些位置的布局必须一致，否则只能插 convert。

然后它做一件很划算的事：

> 试着把**整组**统一到**同一个**布局。统一之后，组内部那些「因为布局不同而产生的 convert」就消失了。

用什么标准衡量「划不划算」？就是这个 pass 一贯的代价模型：统计这次统一后会剩下多少 convert，按循环权重加权。

### 1.2 旧规则的三个分支

```cpp
bool better = adjacentToMoved ? cost.total < bestCost.total
                              : cost.inLoop < bestCost.inLoop &&
                                    cost.total <= bestCost.total;
```

读作：

- **如果这个组件紧挨着「已经改动过的值」**（`adjacentToMoved`）→ 直接比**总成本**，谁小听谁；
- **否则**（组件比较孤立）→ 要求「新方案的**循环内成本**更小」，同时总成本不能变差（`<=`）。

第二条规则的本意是好的：孤立组件多半只影响自己那一片，如果改动只是把循环外的 convert 挪来挪去，那不值得动；只有真能把循环内的 convert 消掉才值得。

### 1.3 坑在「循环内成本」怎么来的

```cpp
static int64_t getScopeLoopFrequencyWeight(Operation *enclosing) {
  constexpr int64_t kLoopIterationWeight = 8;
  constexpr int kMaxWeightedLoopDepth = 3;
  // 数一数有几层 scf.for / scf.while（最多 3 层）
  ...
  return 8^depth;      // 没有循环时返回 1
}
```

```cpp
auto charge = [&](int64_t weight, int64_t c) {
  result.total += weight * c;
  if (weight > 1)             // ← 关键
    result.inLoop += weight * c;
};
```

所以：

- 代码在**循环里** → 权重 ≥ 8 → 这部分成本会计入 `inLoop`；
- 代码**不在循环里** → 权重 = 1 → 只计入 `total`，`inLoop` 一分不加。

于是在一个**完全展开**的 kernel 里（很多算子融合、小循环被完全展开的 kernel 就是这样），组内每一条边都不在循环里：

```
cost.inLoop      = 0
bestCost.inLoop  = 0
```

旧规则的第二个分支就变成 `0 < 0` —— **永远为假**。后果是：

> 无论那个统一布局能消掉多少个边界 convert，都不会被选中。整组保持「每个值各自的原样」，convert 一个都没少。

这不是「优化得不够好」，而是**这条规则在这个场景下完全失效**：它连比较的机会都没有。

### 1.4 一个真实案例

`rlc_ppu` 分支的动机案例是 `scatter_reduce_row_atomic`（64×64）：

- kernel 完全展开，没有外层循环；
- 反向传播阶段（Phase 1b）把 `b[2]` 和 `b[1]` 两种布局都交给了 min/max 那条编码链；
- 冲突消解时，这个组件因为 `inLoop == 0` 而永远打平，最终 atomic 的目标停在默认的 `b[1]` 上，**带着两个桥接 convert**；
- 实际上把整组统一到同一个布局，这两个 convert 都能消掉。

metax 上同类形态（完全展开的归约/散射写回）会遇到一模一样的情形，所以这条修正被一并移植。

---

## 2. 具体做法

在 `better` 计算之后、`if (better)` 之前插入：

```cpp
// 组件周围没有循环时，所有候选的 inLoop 都是 0：
// getScopeLoopFrequencyWeight 在无循环时返回 1，而 charge() 只在 weight > 1
// 时才累计 inLoop。于是上面那条规则变成 0 < 0，永不触发，per-value 的结果
// 就会原封不动留下来，无论统一布局能消掉多少边界 convert。
// 这种情况下也比总成本，但保持严格小于，使真正的平局仍然保留 per-value 结果。
if (!better && !adjacentToMoved && cost.inLoop == 0 &&
    bestCost.inLoop == 0)
  better = cost.total < bestCost.total;
```

逐条读这个条件：

| 条件 | 为什么 |
|---|---|
| `!better` | 旧规则已经判「更好」时不用管 |
| `!adjacentToMoved` | 紧邻已改动值的那条路径本来就在比 `total`，不需要兜底 |
| `cost.inLoop == 0 && bestCost.inLoop == 0` | 只有「双方都没有循环内成本」时，旧规则的比较才是退化的 |
| `cost.total < bestCost.total` | **严格**小于：真平局不选 |

### 2.1 为什么必须是严格小于

如果写成 `<=`，那么「总成本相同」的方案也会被选中——意味着**白改一次布局**。改布局本身不是免费的：它可能让下游多出别的转换、把边界挪个位置。所以这里沿用「只有确实更便宜才动」的原则。

换句话说：

- 旧规则的问题不是「太保守」，而是「根本没比较」；
- 修完之后依然保守：**能便宜就改，持平不改**。

### 2.2 为什么不去改 `charge()` 的累计条件

一个更直觉的「修法」是：让 `charge` 在权重为 1 时也累计 `inLoop`。但那样会破坏原有语义——`inLoop` 的用处就是区分「每一轮都要付的」和「只付一次的」，两者混在一起，`adjacentToMoved` 那条分支之外的权衡就全乱了。

所以选择**在最外层补规则**，而不是动计量口径。这也符合整个增强的风格：宁可加一道局部护栏，不改全局口径。

---

## 3. 走查一个例子

设某组件有 2 个成员，每个成员有 2 个候选布局，且组内所有算子都在函数体顶层（无循环）：

| 统一到 | 组内 convert 剩几个 | `total` | `inLoop` |
|---|---|---|---|
| `b[1,1]/...`（当前各自的结果） | 3 | 96 | 0 |
| `b[1,4]/...`（另一个候选） | 1 | 32 | 0 |

- 旧规则：`adjacentToMoved` 为假 → 比较 `0 < 0` → 假 → **保持现状（3 个 convert）**。
- 新规则：`!better && !adjacent && 0 == 0 && 0 == 0` 成立 → 比较 `32 < 96` → 真 → **换成 `b[1,4]`（1 个 convert）**。

日志（`RLC_DEBUG=2`）：

```
[rlc] rc %atomic_target = tt.atomic_rmw ... minCost=32 candidates=2
        b[1,1]/t[1,32]/w[4,1]/o[1,0] cost=96
        b[1,4]/t[1,32]/w[4,1]/o[1,0] cost=32
[rlc] rc %atomic_target = tt.atomic_rmw ... chosen=b[1,4]/... current=b[1,1]/... tied=1 CHANGED
```

如果两个候选的 `total` 一模一样（真平局），`CHANGED` 不会出现，日志会是 `same`——这是预期行为。

---

## 4. 什么时候生效，什么时候不生效

| 情形 | 是否走新规则 |
|---|---|
| 组件内所有边都不在循环里（`inLoop` 双方均为 0） | **走**（比 `total`） |
| 组件在循环里（`inLoop > 0`） | 不走，由旧规则处理 |
| 组件紧邻「已改动过的值」（`adjacentToMoved`） | 不走（那条路径本来就在比 `total`） |
| 旧规则已判更好 | 不走 |
| 两个候选总成本相等 | 不走（严格小于 → 保持现状） |
| 代价模型被关闭（`FLAGTREE_RLC_ENHANCE_COST_BASED=0`） | 整条组件级消解不运行 |

---

## 5. 局限与边界

1. **只覆盖「双方都无循环」的情形**。如果一个组件的部分值在循环里、部分在外，`inLoop` 会大于 0，仍然由旧规则决定——那种情形下旧规则**是有意义的**（循环内的 convert 更贵），所以不算漏洞。
2. **仍然只比 convert 成本**。这里的 `total` 依旧是「单位成本 × 循环权重」的和，不含寄存器压力、bank conflict、指令调度。它无法判断「convert 少了但访存变差」——那类风险由第 5 篇和 `preservesWritebackMemoryAccess` 等护栏负责，且在**提案生成阶段**就已经拦过了；这里是最后的裁决，面对的是已经通过护栏的候选。
3. **严格小于 → 会错过「同成本但后续更优」的布局**。例如两种布局成本相同，但一种能让后面的 hoist 多消一个 convert。这里不会主动去试。
4. **依赖 `adjacentToMoved` 的语义**。这个量表示「这个组件与已经改过布局的值相邻」，实现上是看成员的算子操作数/使用是否落在 `moved` 集合里。如果组件的划分本身不合理（连通性判断有遗漏），这里的判断也会跟着偏。
5. **不改变正确性**：它只是在若干合法候选之间选一个。
6. **对性能的影响是双向的**：这条规则会让更多组件被统一，从而消掉 convert；但也意味着**更多 IR 被改写**，个别 kernel 上可能引入新的布局边界（虽然总成本是严格下降的）。

---

## 6. 怎么确认它生效了

1. `RLC_DEBUG=1` 先看整体规模：`[rlc] rc layouts=… multi-candidate=…`。
2. `RLC_DEBUG=2` 看具体裁决：`[rlc] rc <值> minCost=… candidates=…` 与 `[rlc] rc <值> chosen=… current=… tied=… CHANGED/same`。
3. 观察重点：
   - 在**没有循环**的 kernel 里，如果看到大量 `same` 且候选成本其实有差异 → 说明新规则**没走到**（检查是否 `adjacentToMoved` 为真，或 `inLoop` 其实不为 0）；
   - 如果看到 `CHANGED` 但总成本差异很小 → 说明平局裁决在边界处工作，属正常。
4. A/B 对照：

```bash
FLAGTREE_RLC_ENHANCE_COST_BASED=0 python bench.py    # 关掉整个代价模型消解做对照
```

注意关掉它**会连带关掉反向传播与小分量求解**（见 `02-phase-switches.md` 的依赖说明），所以差异不全是本功能造成的。想只看本功能，只能靠日志对比 + 代码级判断。

---

## 7. 代码导读

1. `resolveConflicts`——先看组件候选是怎么收集出来的（`componentCandidates`，只在 `enableCostBasedResolution` 时收集）；
2. `resolveConflictsByComponent` 开头——`llvm::EquivalenceClasses` 如何把值连成组件（循环携带值、if 分支、逐元素算子）；
3. `priceAssignment` / `charge`——搞清 `total` 与 `inLoop` 的区别，这是理解本功能的前提；
4. `better` 那三行 + 新增的兜底四行——本体的全部内容；
5. `getScopeLoopFrequencyWeight`——确认「无循环返回 1」。

移植说明：metax 副本删掉了 `#ifdef __PPU__`，使该规则无条件生效；同时删掉了注释里 PPU 专有的案例叙述（`scatter_reduce_row_atomic (64x64, PPU) ...`），保留判定逻辑与「严格小于」的说明。`getScopeLoopFrequencyWeight` 与 `charge` 在 metax 副本中与 ppu 逐字一致，因此这套推理同样成立。
