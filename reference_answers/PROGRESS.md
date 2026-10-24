# 本地完整参考答案：暂停与接续记录

保存日期：2026-10-23（Asia/Shanghai）。**用户明确要求“先到这里为止，保存进度”，当前已暂停；收到继续指令前不再实现或验证。**

## 目标与边界

- 用户授权在本仓库中准备v1～v5完整参考答案，各版文件结构清楚、代码有详细中文注释。
- 每版还需要整体解释：解决的问题、为何这样设计、数学与代码对应、相对上一版变化、验证结果和可优化方向。
- 目录为 `C:/all-demo/ktorch-toy/reference_answers/`，仅本地保存；`.git/info/exclude` 已加入 `/reference_answers/`，不加入Git、不提交、不推送。
- 主项目公开骨架仍保持原状，当前分支仍为main。答案完成度与主项目K01～K10练习进度分开记录。
- 用户允许不验证以减轻本机负担。采用最小编译/小矩阵/短训练检查，逐版串行；不默认执行长训练、大矩阵benchmark或压力测试。未实测内容须据实标注。

## 当前完成情况

| 版本 | 已写入 | 状态及剩余工作 |
| --- | --- | --- |
| v1_minimal | C++二维Tensor/计算图/算子/C ABI；Python Tensor、层、单Block GPT、SGD；文本训练、checkpoint、生成与观察脚本；8个测试 | 已重新编译并通过8项小规模测试（0.043s）；v1 README整体解释已补齐。长训练/性能压测未运行 |
| v2_reliable | 独立复制v1源码；原生AdamW、Python优化器状态、schema2摘要、Trainer、微批梯度平均和训练恢复；9个测试 | 已编译；9项测试通过（0.083s）；CLI已接入SGD/AdamW、batch-size、resume；2步保存schema2、恢复再1步smoke通过。整体README已补齐 |
| v3_cpu | 未创建 | 待实现内核可切换、CPU小规模测量及耗时分解；大规模benchmark默认不执行 |
| v4_inference | 未创建 | 待实现KV Cache、prefill/decode、采样和上下文窗口重建语义 |
| v5_package | 未创建 | 待实现可安装包、CLI、本机HTTP示例及发布工程；必须明确实际测试范围，不直接称企业级 |

顶层README是目标导航，表中的后续版本并不代表那些目录或功能已经完成。

## 已发生的验证（历史证据）

在 `v1_minimal` 下执行过：

1. `python scripts/build.py`：成功，使用 `C:/Program Files/JetBrains/CLion 2025.1.4/bin/mingw/bin/g++.exe`，C++17串行编译。
2. 一个小型直接调用：`a=2, loss=a*a+a` 得到grad=5；完整默认模型参数量912，输入 `[0,1]` 的logits形状为2×4。
3. `python -X utf8 -m unittest discover -s tests -v`：8项通过，报告测试耗时0.033秒。覆盖非方阵matmul与梯度、共享路径/重复gather、稳定CE与因果遮罩、LN/GELU有限差分、模型因果性和临时checkpoint往返、选定完整模型参数梯度、生命周期/no_grad/错误边界，以及25步极小文本训练loss改善。

以上为历史与本次轻量复核结果。没有运行完整100步示例训练，没有记录长期训练性能。v2不是生产级训练框架，仍需进一步完善优化器分组、恢复错误边界和更多专项测试。

没有执行v3～v5验证。

## 当前代码位置和结构

每版的独立源码按以下目录组织：

```text
native/core.hpp       Tensor、算子与梯度接口
native/tensor.cpp     节点、数据版本、反向拓扑与SGD
native/ops.cpp        数学算子的前向和局部反向
native/kernel.cpp     矩阵乘法内核
native/c_api.cpp      uint64句柄、生命周期和错误边界
python/ktorch/        ctypes包装、Tensor、nn、model、optim、checkpoint、training
examples/            train.py / generate.py / inspect_step.py
scripts/build.py      本版独立编译入口
tests/test_core.py     小规模真实行为检查
data/tiny.txt          重复“我爱猫。”的小语料
```

v2新增 `native/adam.cpp`，并扩展ABI、optim.py、checkpoint.py、training.py。两个版本各有自己的LICENSE与源码，不跨版本导入。

## 下一次继续的具体起点

1. 先读取本记录、用户最新要求、根目录AGENTS和本地版本路线；确认Git排除仍生效，公开工作区没有被误改。
2. 优先完成v2的示例训练入口：创建AdamW/SGD和Trainer，支持保存完整训练状态与从checkpoint恢复；明确“继续多少步”的CLI语义。
3. 添加小型恢复等价测试：连续执行N步，与执行A步保存、重载后执行N−A步，比较采样状态、模型参数、优化器矩与步数。再覆盖schema1推理兼容、schema2完整性失败、错误配置和微批平均梯度。
4. 对v1/v2做轻量检查；用户已允许跳过验证，若环境/性能不适合就记录未运行，不启动大任务。
5. 完成v1/v2各自README和整体讲解，尤其解释句柄表而非裸指针、参数身份去重、反向缓存、稳定loss、初始化和局限。部分C++一行仍有多个表达式，可在后续讲解整理时继续展开，不能声称所有行的教学可读性已最终审阅。
6. 在v2稳定后再按版本逐个复制独立源码并实现v3/v4/v5新增能力，逐版更新本记录。

## 续写时需要检查的具体事项

- v2现有 `examples/train.py` 仍调用v1风格 `train()`，不会自动展示新Trainer/AdamW，这是当前最明显的未接通点。
- v2 `checkpoint.py` schema2支持校验和并兼容schema1推理；训练恢复还需专项测试。词表校验顺序、配置总参数量限制、损坏状态字段的错误提示等属于未审阅边界，不要提前声称健壮性全部完成。
- v2微批loss先除batch_size再反向，优化器每批只更新一次；这是二维张量下的梯度累计，并非已经实现3D Tensor批处理。
- 原生SGD/AdamW当前按单参数先计算候选值再写回；整个优化器跨参数的事务原子性尚未承诺。
- 学习绝对位置表下的KV Cache在窗口滑动后必须重新建立与完整窗口重算一致的状态，不能简单保留旧位置编码的缓存。
- v5安装包需包含或构建本平台动态库；仅把Python源码放入wheel不等于可安装可运行。测试干净环境时控制下载和资源。

## 保存时的Git状态

`git status --short --branch` 只显示 `main...origin/main`；公开文件没有未提交修改。`git check-ignore -v reference_answers/README.md` 命中 `.git/info/exclude` 的 `/reference_answers/`。本次答案没有提交、没有推送。

## 2026-10-24 继续后的验证记录

- v1：`python -X utf8 scripts/build.py` 成功；`python -X utf8 -m unittest discover -s tests -v` 运行8项并全部通过，耗时约0.043秒。
- v2：`python -X utf8 scripts/build.py` 成功；同样测试集加AdamW/恢复/摘要完整性用例，运行9项并全部通过，耗时约0.083秒。
- v2 CLI：新训练 `--optimizer adamw --steps 2 --batch-size 2` 成功保存schema2；随后 `--resume --optimizer adamw --steps 1` 成功继续，累计步数2→3，固定窗口loss 1.503775→1.443509→1.418831。独立 `generate.py` 读取checkpoint并输出可用文本。
- v2 Python文件通过标准库 `py_compile` 检查；没有运行大矩阵benchmark、长训练、压力测试或其他平台验证。
- v2剩余：继续检查CLI错误提示、schema1推理兼容、更多恢复等价测试，以及把v3～v5参考答案目录和整体说明补齐。


### 2026-10-24 · versions-v3v5 · v3-v5写入与小规模验证

- v3_cpu已完成：在v1基础上加入可切换C++ GEMM `ijk`/`ikj`策略、benchmark、CPU设计说明和测试。小规模8×8、3次重复结果一致，9项测试通过；只做低负担计时，未做大矩阵压测。
- v4_inference已完成最小KV Cache教学实现：full causal attention与逐token cache逐元素一致，支持snapshot/restore/clear和shape边界；11项继承/缓存测试通过，cache demo最大误差0。尚未把cache嵌入训练模型的每层推理路径。
- v5_package已加入pyproject、CLI、python -m入口、本机标准库HTTP示例、package smoke；10项小测试通过，help和server import通过。未构建wheel、未启动长期服务、未做生产安全/并发验收。
- 三版均用CLion MinGW GCC17编译动态库，采用小矩阵/短流程检查；v3-v5整体README和design.md已写，包含为什么这样设计、边界、优化方向。
- 本次只写reference_answers，未改公开练习源码，未提交/推送；reference_answers继续由.git/info/exclude排除。
- v1/v2状态与整体参考答案进度见本目录PROGRESS.md；v2仍待audit agent接通CLI/AdamW恢复验证。下一步先完成v2再考虑参考答案的统一讲解整理。


## 2026-10-24 继续后的v3～v5与总体验证记录

- v3_cpu已补齐README/design、可切换`ijk`/`ikj` GEMM内核、benchmark和测试。小规模测试：9项通过，约0.043s；`benchmark.py --rows 4 --inner 4 --cols 4 --repeats 1` 两种策略输出最大绝对误差0，示例平均耗时约0.0243ms/0.0193ms。该计时包含Python/ctypes取回，不能外推成生产性能。
- v4_inference已补齐README/design、独立KVCache、完整因果注意力基线、prefill/decode演示和状态测试。小规模测试：11项通过，约0.038s；cache_demo最大绝对误差0.0，cache长度3。没有做大模型延迟压测，没有把cache集成到训练模型的全部层。
- v5_package已补齐README/design、pyproject、CLI、`python -m`入口、本机HTTP server和package smoke。修正Windows子进程帮助输出的UTF-8解码后，10项测试通过，约0.293s；package_smoke通过。没有执行真正wheel构建、干净环境安装、长期HTTP服务、认证/TLS/并发或生产部署检查。
- v1_minimal重新编译并通过8项测试；v2_reliable重新编译并通过9项测试，AdamW 2步保存/恢复1步smoke通过，固定窗口loss 1.503775→1.443509→1.418831。
- 五个版本现在均有独立源码目录和README/design整体解释；答案仍仅存在reference_answers，不加入Git、不提交、不推送。主项目公开源码和K01～K10练习状态没有改变。
- 用户允许不做重验证；本轮只执行小规模检查，不运行长训练、大矩阵benchmark、压力测试或其他平台验证。所有未测范围已在各版README与本记录标明。

## 下一步

完整五版参考答案的当前交付已达到可阅读/可运行的本地参考集合。若继续改进，优先方向是：v2恢复等价测试与CLI边界、v3测量设计、v4把cache接入实际模型并处理窗口位置、v5构建wheel和干净环境安装。除非用户继续指令，不自动扩大范围。
