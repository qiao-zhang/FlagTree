# 功能 2：分阶段开关（以及为什么它们要登记进缓存失效表）

> 对应代码：`envOverride`、`TritonGPURemoveLayoutConversionsPass::runOnOperation`、`include/triton/Tools/Sys/GetEnv.hpp`
> 对应 rlc_ppu commit：`6473c6721`
> 默认状态：全部**开启**

---

## 一句话

把这套布局优化拆成几个可以**分别开关**的阶段，并且能在**不重新编译**的前提下用环境变量临时覆盖——用于灰度、A/B 定位、以及线上兜底。

---

## 1. 为什么需要这个功能

### 1.1 一个真实的两难

这套优化在某类 kernel 上提速明显，在另一类 kernel 上偶有退化。团队想要的不是「全开」或「全关」这种二选一，而是：

- 新功能灰度：先只在一个阶段上开，观察一周；
- 出问题定位：怀疑是「反向传播」那一阶段引入的，就只关它，其余保持；
- 现场兜底：客户机上发现某个模型变慢，用一个环境变量关掉嫌疑阶段，不必等新版本发布。

如果开关是编译期常量，上面三件事都得重新出包；如果是 `pass option`，又得改 Python 侧的 pipeline 代码。所以最终的形态是**三层**：编译期默认 → Python/Knob 传参 → 环境变量临时覆盖。

### 1.2 哪些是开关，哪些是参数

| 层 | 名字 | 性质 | 由谁读 |
|---|---|---|---|
| 主开关 | `enableRlcEnhance` | 总闸 | `knobs.metax.rlc_enhance`（`FLAGTREE_RLC_ENHANCE`，默认开） |
| 阶段 | `enableCostBasedResolution` | 代价模型消解冲突 | pass option，可被环境变量覆盖 |
| 阶段 | `enableBackwardPropagation` | 反向传播 | 同上 |
| 阶段 | `enableSmallComponentSolving` | 小分量求解 | 同上 |
| 阶段 | `enableStoreLayoutRematerialization` | 写回布局重物化 | 同上 |
| 参数 | `FLAGTREE_RLC_ENHANCE_MAX_THREAD_REACH` | 第 5 篇的跨度阈值（字节） | 代码里直接读环境变量，**不是** pass option |
| 参数 | `RLC_DEBUG` | 日志级别 | 代码里直接读环境变量 |

注意最后两行：它们是**数值参数**，不是开关，也不走 `envOverride`。

---

## 2. 具体做法

### 2.1 三层优先级

```
① 主开关 knobs.metax.rlc_enhance        （Python，进程启动时读一次）
        ↓ 与（AND）
② pass option（默认 true，可由 passes.cc 的绑定参数改写）
        ↓ 被环境变量覆盖（仅在环境变量给出了可识别的布尔值时）
③ 最终生效值
```

代码长这样（`runOnOperation` 开头）：

```cpp
bool costBased =
    enableRlcEnhance && envOverride("FLAGTREE_RLC_ENHANCE_COST_BASED",
                                    enableCostBasedResolution);
bool backwardProp = enableRlcEnhance &&
                    envOverride("FLAGTREE_RLC_ENHANCE_BACKWARD_PROPAGATION",
                                enableBackwardPropagation);
bool smallComponentSolving =
    enableRlcEnhance &&
    envOverride("FLAGTREE_RLC_ENHANCE_SMALL_COMPONENT_SOLVING",
                enableSmallComponentSolving);
bool storeLayoutRemat =
    enableRlcEnhance &&
    envOverride("FLAGTREE_RLC_ENHANCE_STORE_LAYOUT_REMAT",
                enableStoreLayoutRematerialization);
```

### 2.2 `envOverride` 的语义

```cpp
static bool envOverride(const char *envVar, bool optionValue) {
  std::optional<bool> envValue =
      tools::isEnvValueBool(tools::getStrEnv(envVar));
  return envValue.value_or(optionValue);
}
```

- `isEnvValueBool` 接受的真值：`on` / `true` / `1`；假值：`off` / `false` / `0`，**大小写不敏感**；
- 环境变量**未设置**、或写了无法识别的字符串（例如 `yes`、`enable`）→ `isEnvValueBool` 返回空，退回 pass option 的值；
- 空字符串同理（`getStrEnv` 未设置时返回 `""`）。

这条设计的意义：**写错不会静默变成「关」**。很多项目的布尔环境变量是「非空即真」，一旦有人写 `FLAGTREE_RLC_ENHANCE_COST_BASED=no` 就会误开/误关；这里写错等于没写，行为退回编译期默认，容易发现。

### 2.3 关键：阶段之间有依赖

```cpp
if (!costBased) {
  backwardProp = false;
  smallComponentSolving = false;
}
```

- **代价模型关 → 反向传播和小分量求解必须一起关**。原因很实际：这两个阶段会产生「提案」，而验收提案的正是代价模型；没有验收闸门就改布局等于**无脑套用**提案，风险远大于收益。
- **写回重物化（storeLayoutRemat）不受这条约束**：它是独立的一套 remat 逻辑，不依赖代价模型。
- 主开关为假 → 四个全为假，pass 完全回到原始（未增强）行为。

这条依赖关系是「为什么关一个却像关了三个」的根源，排查时最容易踩。

### 2.4 为什么必须登记到 `CACHE_INVALIDATING_ENV_VARS`

这一条不是「顺手加的整洁」，缺了会真的坏：

**原因一：不登记会在 debug 构建里断言失败。**
`include/triton/Tools/Sys/GetEnv.hpp` 里的 `getStrEnv` / `getBoolEnv` 第一行就是 `assertIsRecognized(env)`。它检查变量名是否在 `CACHE_INVALIDATING_ENV_VARS` 或 `CACHE_NEUTRAL_ENV_VARS` 里；不在的话在带断言的构建里直接中止。所以新增环境变量必须登记，否则本机调试能过、CI 或 release 之外的构建反而崩。

**原因二：不登记会让缓存给出「看似没生效」的结果。**
Python 侧在计算 kernel 的磁盘缓存键时会调用 `get_cache_invalidating_env_vars()`（见 `python/triton/compiler/compiler.py`、`python/triton/runtime/jit.py`、`python/triton/runtime/autotuner.py`）。这个函数读的就是 `CACHE_INVALIDATING_ENV_VARS`。

如果不登记：你把 `FLAGTREE_RLC_ENHANCE_COST_BASED=0` 打开跑一次，编译器发现「缓存键没变」，直接复用上次编译好的二进制——**你以为开关没效果，其实是压根没重新编译**。这个坑非常隐蔽。

于是这几个变量都进了表：

```cpp
"FLAGTREE_RLC_ENHANCE_COST_BASED",
"FLAGTREE_RLC_ENHANCE_BACKWARD_PROPAGATION",
"FLAGTREE_RLC_ENHANCE_SMALL_COMPONENT_SOLVING",
"FLAGTREE_RLC_ENHANCE_STORE_LAYOUT_REMAT",
"FLAGTREE_RLC_ENHANCE_MAX_THREAD_REACH",
```

---

## 3. 什么时候读、读几次（容易踩的细节）

| 变量 | 读取时机 | 同一进程内改环境变量有效吗 |
|---|---|---|
| `FLAGTREE_RLC_ENHANCE` | Python 进程启动、`knobs` 初始化时 | **无效**（Knob 是启动时快照） |
| 四个阶段开关（经 `envOverride`） | **每次** `runOnOperation` 执行时 | 有效（下一个被编译的 kernel 会看到新值） |
| `FLAGTREE_RLC_ENHANCE_MAX_THREAD_REACH` | 第一次用到时的 `static const` 初始化 | **无效**（进程内只读一次） |
| `RLC_DEBUG` | 第一次用到时的 `static const` 初始化 | **无效**（进程内只读一次） |

实践建议：

- 做 A/B 时**改环境变量后重启进程**最稳，可以避免上面「static 只读一次」带来的困惑；
- 同一进程内想切换阶段开关，注意已编译过的同一个 kernel 会命中磁盘缓存，需要改输入（或清缓存）才会重新编译。

---

## 4. 一个走查例子

场景：怀疑「小分量求解」在某模型上导致退化，想只关它。

```bash
# 基线
python bench.py                      # 记录耗时与生成物

# 只关小分量求解：注意代价模型必须保持开，否则会连带关掉别的阶段
FLAGTREE_RLC_ENHANCE_SMALL_COMPONENT_SOLVING=0 python bench.py

# 顺带确认它到底有没有被跳过：小分量阶段的日志应该消失
FLAGTREE_RLC_ENHANCE_SMALL_COMPONENT_SOLVING=0 RLC_DEBUG=1 python bench.py 2>&1 | grep 'entry='
```

第三步很关键：如果 `entry=reduceTail / loopTail / storeTail / producerSide` 这几行还在，说明环境变量没被读到（多半是写错了值，或者恰好命中了缓存）。见第 5 篇与第 4 篇文档的日志说明。

反向验证也常用：

```bash
# 全关，确认性能回到「没有这套优化」的水平，用来确认优化确实在起作用
FLAGTREE_RLC_ENHANCE=0 python bench.py
```

---

## 5. 局限与边界

1. **粒度只到「阶段」**，到不了「单个检查」。例如想单独关第 3 篇的「别搬走消费者」，只能连整个小分量求解一起关。
2. **主开关不能中途改**（Knob 语义），必须新进程。
3. **两个数值参数是进程内只读一次的**（`static` 缓存），改了要么重启进程，要么接受不一致。
4. **关掉代价模型会连带关掉两个阶段**，这既是安全设计也是排查陷阱。
5. **环境变量进入缓存键**意味着：频繁在同一个 `TRITON_CACHE_DIR` 里切换开关会**积累多份缓存的二进制**，磁盘占用上升。这是换取「改开关即生效」的代价。
6. **环境变量覆盖只影响这个 pass**。pipeline 里 `remove_layout_conversions` 被调用多次（`make_ttgir` 里至少 6 处），每处都读同一个环境变量、都受同样覆盖，无法「只让第 2 次调用用新行为」。
7. **写错的值不报错**（按「未设置」处理），这是刻意的取舍，但也意味着拼写错误不会被提醒——务必用 `RLC_DEBUG` 或日志确认。

---

## 6. 怎么确认开关生效了

| 现象 | 说明 |
|---|---|
| `RLC_DEBUG=1` 里出现 `entry=reduceTail / loopTail / storeTail / producerSide` | 小分量求解**在跑** |
| 上面四行消失，但 `[rlc] cost before=` 仍出现 | 小分量求解被关，代价模型还在（其他路径仍会用它） |
| `[rlc] rc layouts=` 仍出现 | 冲突消解阶段在跑（它总在跑，但内部行为随 `costBased` 变化） |
| 什么 `[rlc]` 都没有 | `RLC_DEBUG` 没被读到（未设置/值为 0/进程内已缓存过初值），或主开关关着 |

怀疑「开关没效果」时的排查顺序：

1. `env | grep FLAGTREE_RLC_ENHANCE` 确认拼写（应能看到你设的那个阶段变量）；
2. 确认值能被 `isEnvValueBool` 识别（`on/true/1` 或 `off/false/0`）；
3. 关掉磁盘缓存再跑：`TRITON_ALWAYS_COMPILE=1`（或换 `TRITON_CACHE_DIR`）；
4. 确认进程是新起的（主开关与两个数值参数有启动期快照）。

---

## 7. 代码导读

1. `include/triton/Tools/Sys/GetEnv.hpp`
   - `CACHE_INVALIDATING_ENV_VARS`（登记表）、`CACHE_NEUTRAL_ENV_VARS`（不影响缓存的变量）
   - `assertIsRecognized`（为什么必须登记）
   - `getStrEnv` / `getBoolEnv` / `isEnvValueBool`（取值与解析规则）
2. `envOverride`（`RemoveLayoutConversions.cpp`，`hoistConvert` 之后）
3. `TritonGPURemoveLayoutConversionsPass::runOnOperation`（优先级与依赖关系）
4. `third_party/metax/python/src/passes.cc`（pass option 的 Python 绑定与默认值）
5. `third_party/metax/backend/compiler.py` 的 `make_ttgir`（主开关如何传进来）
6. `python/triton/knobs.py` 的 `metax_knobs.rlc_enhance`

移植说明：这一份是**纯共享代码**，metax 与 ppu 完全一致，没有 `__PPU__` 差异。
