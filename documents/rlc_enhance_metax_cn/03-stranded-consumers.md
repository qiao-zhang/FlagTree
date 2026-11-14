# 功能 3：别把 load 的消费者「搬走」（避免白搬一趟）

> 对应代码：`LayoutPropagation::solveSmallComponents` 里的 `commitIfProfitable`（`producerKeepsItsLayout`）
> 对应 rlc_ppu commit：`f0513d863`
> 默认状态：**开启**，受 `FLAGTREE_RLC_ENHANCE_SMALL_COMPONENT_SOLVING` 控制

---

## 一句话

如果一个提案想把「一个昂贵 load 的下游」全搬到新布局、却动不了这个 load 本身，那就一定会多出一个 convert——而这个 convert 恰好落在提案声称要消掉的那条边上。所以**直接否决**这种提案。

---

## 1. 问题长什么样

### 1.1 用搬家打个比方

设想搬家公司的报价单写着「把 A 房间的东西搬到 B 房间，省下搬运费」。但实际情况是：

- A 房间里有个**焊死在地上的大柜子**（昂贵 load，布局不能改）；
- 搬家公司动不了柜子，只能把**柜子周围的所有东西**搬到 B，然后在柜子和 B 之间加一段**临时通道**（新的 `convert_layout`）；
- 而这段临时通道，恰恰就是报价单上说「要拆掉的那条旧通道」。

结果：旧通道没拆掉，又多修了一条。钱多花了，路更长了。

### 1.2 具体到这张图

```
   [ 昂贵 load ]  ← 布局不能改（sizePerThread=[4]，为了访存连续）
        |
        v
    （一串算术：加减、取整、比较……）        ← 提案想把这一串全部改成 [1]
        |
        v
   [ 某个 store / 归约 / gather ]           ← 用来验收的 convert 在更下游
```

提案的动机是「这一串值和下游布局不一致，统一之后能消掉一个 convert」。它算的账是：

> 把这一串值改成布局 X，能消掉 N 个 convert。

但它漏算了：**这一串的源头是一个不能改布局的昂贵 load**。一旦这一串改成 X，load 与这一串之间就必须插一个 `load布局 → X` 的 convert。于是真实账单是：

> 消掉 N 个，但新增 1 个（可能还连带新增下游的其它 convert），净值 ≤ -1。

`rlc_ppu` 分支上的原始复现案例（`coalesce scatter`）就是这样：`%k_22 = tt.load ... sizePerThread=[4]` 供给整条 `pos` 计算链，提案把这条链改成 `[1]`，结果**保留了 `[4]→[1]` 的 convert，还额外新增了两个**（gather 结果、vals load），从 0 个变成 3 个。

### 1.3 为什么原来的验收挡不住

验收用的是代价模型（`shouldAcceptProposal`）：只统计提案涉及的值的**直接使用边**上的 convert 成本，比较「改前 vs 改后」。问题在于：

- load 的那条边，在提案看来「不属于我改的范围」，容易被漏计；
- 提案把「能消掉的 convert」算得很实，而「新长出来的 convert」如果不在它的视野里，账就是虚的。

所以这里不是「调整权重」能解决的，而是**视野问题**：需要在提案落地前，抬头看一眼上游是不是有个动不了的昂贵 load。

---

## 2. 具体做法

在 `commitIfProfitable` 里，过了「消掉的 convert 数 > 0」这一关之后，插入一段检查：

### 2.1 第 1 步：把提案里每个值的操作数都翻一遍

```cpp
for (auto &it : proposal) {
  Operation *defOp = it.first.getDefiningOp();
  if (!defOp)
    continue;                        // 块参数（循环携带值）没有定义算子，跳过
  for (Value operand : defOp->getOperands()) {
    if (proposal.count(operand))
      continue;                      // ① 也在提案里：一起改，已经被计价
    if (!isa<RankedTensorType>(operand.getType()))
      continue;                      // 非张量，无关
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    if (!operandType->getEncoding() ||
        operandType->getEncoding() == it.second)
      continue;                      // ② 没布局，或本来就和目标一致
    if (producerKeepsItsLayout(operand))
      return false;                  // ③ 上游死守着布局 → 否决整个提案
  }
}
```

三个「跳过」都很重要，它们决定了这个检查**不会**过度开火：

- ① **提案内部的值**：这些值会被一起改写，边上不会残留 convert，代价模型已经算过；
- ② **布局本来就一致的操作数**：改过去不需要桥接，不产生新 convert；
- ③ 只有当操作数「有布局、且与目标布局不同」时，才需要问「它能不能跟着动」。

### 2.2 第 2 步：问「它能不能跟着动」

```cpp
auto producerKeepsItsLayout = [&](Value operand) {
  constexpr unsigned kMaxConvertHops = 4;       // 最多往下钻 4 层 convert
  for (unsigned hop = 0; hop < kMaxConvertHops; ++hop) {
    Operation *def = operand.getDefiningOp();
    if (!def)
      return false;
    if (auto cvt = dyn_cast<ConvertLayoutOp>(def)) {
      operand = cvt.getSrc();                   // 穿过 convert 继续往下
      continue;
    }
    return isa<LoadOp, StoreOp>(def) && isExpensiveLoadOrStore(def);
  }
  return false;                                  // 钻了 4 层还没到底 → 不认为有问题
};
```

要点：

- **只穿过 `convert_layout`**。遇到任何其它算子就停下并回答「不是昂贵 load」，因为其它算子通常可以被重标（这就是小分量求解的常规工作）；
- **`kMaxConvertHops = 4`**：提案里的值往往挂在 load 下游好几个算术节点之后，中间常夹着若干 convert/cast。4 层足够覆盖常见链条，又不至于把整张图翻一遍；
- 钻到第 4 层仍未到达 load → **认为没问题**（放行）。宁可少拦，也不要因为「图太深看不透」就把合法提案全否掉。

### 2.3 为什么这条检查放在这个位置

它位于「消掉的 convert 数已经 > 0」之后、「其它否决条件」之前。也就是说：

- 提案连一个 convert 都消不掉时，前面就已经否决了，不用走到这里；
- 这条检查**只否决、不择路**：它不会说「那换个布局试试」，只是把当前这个提案扔掉，后面的候选编码还有机会。

---

## 3. 走查一个例子

设提案内容是：把 `%a、%b、%c` 三个值统一成 `b[1,1]/t[1,32]/w[4,1]/o[1,0]`（即每线程 1 个元素）。

```
%ld  = tt.load %ptr        : b[1,4]/...       ← 昂贵 load，sizePerThread=[4]
%cvt = ttg.convert_layout %ld : b[1,1]/...    ← 现有桥接 convert
%a   = arith.addf %cvt, %x
%b   = arith.mulf %a, %cvt
%c   = arith.cmpf ... %b
```

- 检查 `%a` 的操作数：`%cvt` 在提案里吗？不在。它的 encoding 是？与目标一致（`b[1,1]`）→ 跳过（第 ② 条）。
- 检查 `%b` 的操作数：同上，跳过。
- 结论：**这条提案被放行**。为什么？因为这里 load 与提案链之间**已经存在** convert，提案并没有让它新增——它消的是更下游的 convert。

再看另一种形态：

```
%ld = tt.load %ptr : b[1,4]/...    ← 昂贵 load
%a  = arith.addf %ld, %x           ← 直接消费 load，中间没有 convert
%b  = arith.mulf %a, %c
%c  = ttg.convert_layout %b : b[1,1]/...   ← 想消掉这个
```

- 提案把 `%a、%b` 改成 `b[1,1]`。
- 检查 `%a` 的操作数 `%ld`：不在提案里、encoding 是 `b[1,4]` ≠ 目标 `b[1,1]`。
- `producerKeepsItsLayout(%ld)`：没有 convert 可穿，直接命中「昂贵 load」→ 返回真。
- **提案被否决**。理由正确：真要这么改，`%ld → %a` 之间会长出一个新 convert，而 `%c` 那个 convert 只是被挪了个位置。

---

## 4. 什么时候生效，什么时候不生效

| 条件 | 是否检查 |
|---|---|
| 上游是**昂贵** load/store（`isExpensiveLoadOrStore`）且布局不同 | **否决** |
| 上游是便宜 load（单元素、或元素数少于线程数） | 不否决（它本就可以跟随重标） |
| 上游与目标布局一致 | 不否决 |
| 上游在提案里（会被一起改） | 不否决 |
| 上游是普通算术算子 / 块参数 | 不否决（不是本检查的职责） |
| 中间隔着超过 4 层 convert 才到 load | 不否决（超出探测深度） |
| 提案连一个 convert 都消不掉 | 更早的地方就否决了 |

---

## 5. 局限与边界

1. **只看「直接操作数」，不追整条数据流**。如果 load 隔了 5 层算术而不是 convert，本检查看不到（那属于代价模型和别的护栏的职责）。
2. **4 层 convert 是经验深度**。它假设「提案值到 load 之间最多夹 4 个 convert」；更深的链条会被放过。这个数字是常量，改动需同时评估误杀率。
3. **只认昂贵 load/store**。这是刻意的：便宜 load 跟随重标的代价小，不值得拦。
4. **只否决，不记账**。「新增 convert 的精确数量」仍然交给代价模型；本检查是个**粗筛**，专治「明显亏」的那一类。
5. **不区分「新增 1 个换消掉 1 个」这种临界情形**。严格来说「净变化为 0」也可能因为 convert 的位置（循环内/外）而有不同代价；这里一律否决，偏保守。
6. **对块参数（循环携带值）不做判断**：这类值没有定义算子，检查直接跳过。所以「循环迭代值被搬走、而它来自循环外的昂贵 load」这种形态落在视野之外。
7. **没有正确性风险**：被否决只是保留原布局。

---

## 6. 怎么确认它生效了

`RLC_DEBUG=1`（或 `2`）时会打印：

```
[rlc] reject untaggable-load-boundary: <值> <- <操作数> with <操作数布局> vs <目标布局>
```

例如：

```
[rlc] proposal n=3
        %a = arith.addf ... -> b[1,1]/t[1,32]/w[4,1]/o[1,0]
[rlc] proposal removed=1
[rlc] reject untaggable-load-boundary: %a = arith.addf ... <- %ld = tt.load ... with b[1,4]/t[1,32]/w[1,1]/o[1,0] vs b[1,1]/t[1,32]/w[4,1]/o[1,0]
```

读法：提案想把 `%a` 改成每线程 1 个元素的布局，但它的操作数 `%ld`（一个 load，当前每线程 4 个连续元素）在上面，两者的布局不同 → 否决。

排查建议：

- 如果这条日志出现得很频繁、而你确信那些提案本来能赚，可以用 `FLAGTREE_RLC_ENHANCE_SMALL_COMPONENT_SOLVING=0` 做对照，看 TTGIR 差异和性能差异；
- 这条日志**只在 `RLC_ENHANCE` 打开且小分量求解在跑时才有意义**，其余情况看不到属正常。

---

## 7. 代码导读

1. `solveSmallComponents` → `commitIfProfitable`：先看 `if (removedConverts == 0)` 和它后面的注释块，那是本检查的落点；
2. `producerKeepsItsLayout`：十来行的 lambda，先看它，再看外层的双重循环；
3. `isSmallComponentRetaggable`：理解「为什么 load 不能跟随」——它明确拒绝了 `isa<LoadOp, StoreOp>(defOp) && isExpensiveLoadOrStore(defOp)` 的值；
4. `isExpensiveLoadOrStore`（`lib/Dialect/TritonGPU/Transforms/Utility.cpp`）：昂贵 load 的定义。

移植说明：metax 副本删掉了 `#ifdef __PPU__`，使其无条件生效；同时删掉了注释里 PPU 专有的案例描述（`coalesce scatter, PPU: ...`），理由写在文件里保留下来的注释中。`rlc_ppu` 原始 commit 还顺带修正了一处 `return false;` 的缩进，metax 分支那处本来就是正确缩进，因此无需该改动。
