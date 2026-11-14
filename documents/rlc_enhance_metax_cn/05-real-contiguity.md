# 功能 5：布局宽度 ≠ 真实地址连续性（真实连续性门控）

> 对应代码：`widensPastRealContiguity`、`findAddressAxisInfo`、`findAddressPitch`、`rlcMaxThreadReachBytes`
> 对应 rlc_ppu commit：`65eb83320`
> 默认状态：**开启**；阈值可用 `FLAGTREE_RLC_ENHANCE_MAX_THREAD_REACH` 覆盖

---

## 一句话

布局上说「每个线程连续拿 4 个元素」，不等于内存里那 4 个地址真的挨在一起。当地址不连续、而这次改动又要求更宽的访问时，**保留原来的 `convert_layout`**——它是片上的数据重排，比一趟虚张声势的宽访问便宜。

---

## 1. 问题长什么样

### 1.1 一条容易骗过眼睛的捷径

判断「改了布局之后访存会不会变差」，代码里原先有两道检查：

- `preservesWritebackMemoryAccess(...)`：比较新旧布局在**内存序方向**上每线程连续的元素数，要求「不能变小」；
- `widensPastVectorAccess(...)`：如果宽度已经达到 128 位（`kMaxAccessBits`）的向量宽度，就不允许再无节制加宽。

两者都依赖同一个量：`getContigAlongMemoryOrder(type)`——「沿内存最快维，一个线程手里真正连续的元素数」。而这个量来自 `getOrderForMemory(type)`，后者有一个兜底规则：

> 如果布局的**最快维是一个大小只有 1 的退化维**（长度 1 的维度上谈不上连续性），就回退到「线程顺序（thread order）」来判断。

于是麻烦出现了：在这种退化情形下，两个布局的「内存序方向」其实不是同一件东西，比较退化成**纯布局宽度的比较**。「新布局每线程 4 个 ≥ 旧布局每线程 1 个」→ 检查通过 → 允许改。

但真实地址可能是**跳着**的。

### 1.2 一个直观的例子：转置访问

看一个 `aminmax(dim=0)`（沿列归约）时的拷贝 kernel：

- 要读的第 0 维在内存里是**跨行**的，行距等于行长；
- 每个线程读到的元素，彼此间隔「一整行」，属于散点访问；
- 旧布局：`sizePerThread=[1,1]`、`order=[0,1]`（线程沿第 0 维排，一次读 1 个散点）＋ 一个 `convert_layout` 把结果整理给归约；
- 「优化」方案：把 load 的布局改成 `sizePerThread=[1,4]`、`order=[1,0]`，消掉那个 convert。

账面上「少了一个 convert」。实际上：

- 每个线程原本发 **1 个**散点请求，现在要发 **4 个跨行**的散点请求（硬件把它们合成一个「跨 4 段」的宽请求）；
- 被消掉的 convert 只是**片上**的一次重排。

实测（PPU 上的标定数据）：小规模 `[4096, 4096]` 转置里 convert 占主导，这个改法 **+10.5%**；但 stride 65536 时 **-7.3%**、fp32 且 1048576 时 **-19.4%**。同一种改法、同一个 kernel 家族，规模一变就由赚变赔——因为决定胜负的是**地址有多散**，而不是 convert 有几个。

### 1.3 为什么不能靠「调权重」解决

代价模型只认识 convert 的个数与循环权重，它根本不知道「这次 load 的地址是跳着的」。所以必须在**动布局之前**，抬头看一眼地址分析（AxisInfo）给出的真实连续性。

---

## 2. 具体做法：四步判定

入口是 `widensPastRealContiguity(axisInfo, ptr, elementType, targetEncoding)`。它只在「确实要变宽」时才开始细算，其余情况一律放行。

```
① 前置筛（任一不满足就放行）
     - 有地址分析（axisInfo 非空）
     - 目标布局是 blocked（不是 MMA / linear 等）
     - 目标布局 ≠ 当前布局
     - target > current（确实要求每线程更宽）
        ↓ 通过
② 找真实连续性
     realContig = min(地址分析给出的连续性, 该维长度)
     - target <= realContig  → 地址真的够宽 → 放行
     - 找不到地址分析        → 保守拒绝
        ↓ 不够宽
③ 找行距（pitch）
     - 找不到 pitch          → 保守拒绝
        ↓ 找到
④ 估算「一个线程这次最远跨多少字节」
     reach = target × pitch × 元素位宽 / 8
     - reach > 阈值(默认 48 KiB) → 拒绝
     - 否则                      → 放行
```

### 2.1 第 ① 步为什么要「目标必须是 blocked」

这道检查守的是 `sizePerThread`（每线程连续元素数）。而 MMA（tensor core）布局的每线程 tile 是由矩阵乘的形状决定的，不是由地址模式决定的——拿地址连续性去衡量它没有意义。所以非 blocked 目标直接跳过。

### 2.2 第 ② 步为什么需要「往上游走几层」

```cpp
static AxisInfo *findAddressAxisInfo(ModuleAxisInfoAnalysis *axisInfo,
                                     Value value, unsigned depth = 3);
```

- 地址分析是按值建表的，只对它「看过」的值有记录；
- 而这个阶段会在运行过程中重写 IR，**新造出来的地址链可能还没进分析表**，尽管它的操作数在；
- 所以这里沿操作数往下探最多 3 层：`addptr` 是「基址 + 偏移」，它的操作数携带的连续性，正是结果会有的连续性。

拿到之后再取实际：

```cpp
unsigned dim = targetOrder.front();          // 目标布局的内存序第一维
realContig = std::min(info->getContiguity(dim), ptrType.getDimSize(dim));
```

`min` 里的第二项是「这一维本来就这么长」，防止把「维长」误当成「可连续的宽度」。

判定：`target <= realContig` → 地址真的支持这么宽 → 放行。否则继续第 ③ 步。

**找不到地址分析时为什么拒绝**：这说明这条地址链在该阶段没有任何连续性信息可用。既然「更宽的访问」本来就只是为了替代一个片上 convert，而在信息缺失的情况下更宽的访问有放大风险，就选择**保留 convert**。

### 2.3 第 ③④ 步：用行距估「跨了多远」

地址不连续时，宽请求会被拆成多段。拆几段本身不致命，致命的是**段与段之间隔多远**：

- 隔得近（同一行/同一页附近）→ 硬件还能凑效，比一个 convert 强；
- 隔得远（跨越多个页）→ 每次宽请求都在制造 page 级放大。

代码里唯一能静态看到的「隔多远」信号，是跨步地址的构造方式 `index * pitch` 里的常量：

```cpp
static std::optional<int64_t> findAddressPitch(Value value, unsigned depth = 6);
```

它往下探最多 6 层，遇到 `arith.muli` 时取那个常量操作数作为行距（跳过 0）。找不到就返回空 → **保守拒绝**（没有可估的量，就认为不该赌）。

拿到 pitch 之后：

```
reach = target(每线程连续元素数) × pitch(行距) × 位宽 / 8
```

即「一个线程这次宽访问最远跨多少字节」。与阈值比较：

```cpp
static int64_t rlcMaxThreadReachBytes() {
  static const int64_t limit = [] {
    const char *env = std::getenv("FLAGTREE_RLC_ENHANCE_MAX_THREAD_REACH");
    int64_t raw = env ? std::strtoll(env, nullptr, 10) : 0;
    return raw > 0 ? raw : int64_t(48) * 1024;   // 默认 48 KiB
  }();
  return limit;
}
```

- 默认 **48 KiB**，来自 PPU 的实测标定：reach ≤ 32 KiB 时普遍是收益（+4.9% ~ +10.5%），≥ 64 KiB 时普遍退化（-1.2% ~ -19.4%），48 KiB 取在其间；
- 环境变量给正整数可覆盖；给了 0 或负数视为未设置（回到默认）；
- **进程内只读一次**（`static`），改值需重启进程。

### 2.4 两个调用点

| 调用点 | 场景 | 需要的地址分析从哪来 |
|---|---|---|
| `collectProducerClosure` 的 load 分支 | 前向收集 / 小分量求解要在改一个昂贵 load 的布局 | `LayoutPropagation` 构造时传入；**只有在 `smallComponentSolving` 开启时才会构建** |
| `isDuplicableLoadInEncoding` | remat 阶段要复制这个 load、用目标布局重新 load | `backwardRematerialization(module, duplicableLoadRemat)` 在 `duplicableLoadRemat` 为真时自行构建；pass 里传入的是 `storeLayoutRemat` |

这一点很关键，容易踩：**如果只关掉「小分量求解」，闭包路径上的这道检查就因为没有地址分析而一律放行**（走 `skip("no-analysis")` 分支）。它不会误拒，但也就不再保护。

---

## 3. 走查一个例子

设目标布局要求每线程连续 4 个元素（`target = 4`），元素为 fp32（4 字节 = 32 位），行距 `pitch = 4096`：

```
realContig = min(地址分析连续性=1, 该维长度=4096) = 1
target(4) > realContig(1)                  → 地址不够宽，继续
pitch 找到 = 4096
reach = 4 × 4096 × 32 / 8 = 65536 字节 = 64 KiB
64 KiB > 48 KiB                            → 拒绝
```

日志（`RLC_DEBUG=2`）：

```
[rlc] real-contiguity ptr=%ptr tgt=b[1,4]/... cur=1 tgt=4 real=1 pitch=4096 reach=65536 limit=49152 -> reject
```

再换成 `pitch = 512`：

```
reach = 4 × 512 × 32 / 8 = 8192 字节 = 8 KiB ≤ 48 KiB  → 放行
```

同样的布局改动，只因为行距小，就从「拒绝」变「放行」。这就是这个功能想表达的东西：**决定权在地址隔多远，不在 convert 省几个**。

---

## 4. 什么时候生效，什么时候不生效

| 条件 | 行为 |
|---|---|
| 目标布局不是 blocked（MMA/linear 等） | 跳过（放行） |
| 目标布局等于当前布局，或并未要求更宽 | 跳过（放行） |
| 没有地址分析 | 跳过（放行）——注意这一条是「不保护」 |
| 目标 pointer 不是张量 / 元素类型非整非浮点 | 跳过（放行） |
| 地址真的连续到 `target` 宽 | 放行 |
| 地址链里找不到任何连续性信息 | **拒绝**（保留 convert） |
| 地址不连续且找不到行距 | **拒绝** |
| 地址不连续、行距已知、`reach ≤ 48 KiB` | 放行 |
| 地址不连续、行距已知、`reach > 48 KiB` | **拒绝** |

---

## 5. 局限与边界（这一节最重要）

1. **只看内存序第一维**（`targetOrder.front()`）。多维同时变宽、或真正的瓶颈在第二维时，判断可能不全面。
2. **`reach` 是上界式的一维估算**：`每线程连续数 × 行距 × 位宽`。它不含页大小/对齐/TLB 的精确模型，也不知道运行期才知道的 extent（比如矩阵实际有多少行是运行时参数）。它只是一个「数量级」判据。
3. **行距是被「猜」出来的**：从 `index * 常量` 里取那个常量。如果地址用别的形式构造（例如先乘后加已被折叠、或用运行期 pitch），就找不到，于是走**保守拒绝**——这会让一些本该赚的提案被挡掉。
4. **48 KiB 是 PPU 的标定值**，metax 沿用同一默认。metax 的页大小、访存通路、行距代价都可能不同，建议在 metax 上用 `FLAGTREE_RLC_ENHANCE_MAX_THREAD_REACH` 做 3~4 档 A/B（如 16 / 48 / 128 KiB）后固定下来。
5. **阈值与日志级别都是进程内只读一次**（`static`），改值要重启进程；同时也注意它们进了缓存键，改值会触发重新编译。
6. **依赖地址分析是否存在**。如前所述，关掉「小分量求解」会让闭包路径上的这道检查形同不存在（`axisInfo == nullptr` → 放行）。这不是 bug，但会让人误以为「检查失效了」。
7. **只覆盖「读」的一侧**。写回（store/atomic）走的是 `preservesStoreMemoryAccess` / `writeback` 系列逻辑，有自己的判据，不在本功能范围内。
8. **保守的代价**：信息缺失时一律拒绝 → 可能放过一些真实收益。若确认自己被过度保守地挡住（`RLC_DEBUG=2` 里频繁出现 `reject(no-axisinfo)` / `reject(no-pitch)`），可先把阈值放大做对照实验，但**不建议长期关掉**：这一类误判的代价通常是两位数百分比的性能倒退。
9. **无正确性风险**：拒绝只是保留原布局。

---

## 6. 怎么确认它生效了

`RLC_DEBUG=2` 下有一整套标签：

| 日志 | 含义 |
|---|---|
| `[rlc] real-contiguity skip(no-analysis)` | 没有地址分析，未检查 |
| `skip(no-target-or-element)` | 目标布局为空或元素类型不支持 |
| `skip(not-tensor-ptr)` | 指针不是张量 |
| `skip(target-not-blocked)` | 目标是 MMA/linear 布局，跳过 |
| `skip(target-is-current)` | 目标就是当前布局 |
| `skip(no-widen)` | 没有要求更宽（`cur=` / `tgt=` 给出两个宽度） |
| `skip(no-order)` | 取不到内存序 |
| `reject(no-axisinfo)` | 地址链里没有任何连续性信息 → 拒绝 |
| `allow(coalesced)` | 地址真的够连续 → 放行 |
| `reject(no-pitch)` | 地址不连续且找不到行距 → 拒绝 |
| `real-contiguity ptr=… cur=… tgt=… real=… pitch=… reach=… limit=… -> reject/allow` | 完整判定过程与数字 |

闭包路径上的否决还会额外打印：

```
[rlc] reject widen-past-real-contiguity: %ptr -> b[...]
```

remat 路径上是：

```
[rlc-remat-load] reject widen-past-real-contiguity: %ptr -> b[...]
```

**推荐排查动作**：把 `reach` 与 `limit` 两个数抄下来。如果大量 `reach` 只是略微超过 `limit`，说明阈值偏低；如果 `reach` 是 `limit` 的几十倍，那这些拒绝是对的。

```bash
# 试试更大/更小的阈值（记得重启进程）
FLAGTREE_RLC_ENHANCE_MAX_THREAD_REACH=131072 RLC_DEBUG=2 python bench.py 2>&1 | grep 'real-contiguity'
```

---

## 7. 代码导读

按这个顺序读，理解成本最低：

1. `getContigAlongMemoryOrder` / `getOrderForMemory`（文件内 + Triton 公共库）——先搞清「布局上的宽度」是怎么算的，以及它在退化维上的兜底规则；
2. `preservesWritebackMemoryAccess`、`widensPastVectorAccess`——看清原有两道门为什么挡不住（注意 `widensPastVectorAccess` 里的 `kMaxAccessBits = 128`）；
3. `findAddressAxisInfo`（很短，只是找表项）；
4. `findAddressPitch`（找行距，注意它只认 `arith.muli`，且跳过 0）；
5. `rlcMaxThreadReachBytes`（阈值读取）；
6. `widensPastRealContiguity` 本体——四步判定依次排开；
7. 两个调用点：`collectProducerClosure` 的 load 分支、`isDuplicableLoadInEncoding` 末尾。

移植说明：metax 副本删掉了 `#ifdef __PPU__`，使这道检查无条件生效；注释里 PPU 的标定案例保留但明确标注为「移植来源」，并写明 metax 沿用同一默认值、可通过环境变量重标定；此外没有改动任何判定逻辑或常量。
