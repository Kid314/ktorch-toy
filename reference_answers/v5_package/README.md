# v5：可安装、可调用的开源工程参考答案

v5 将训练出的模型整理成别人能够调用的工程：标准 `pyproject.toml`、Python 模块入口、生成命令和一个只绑定本机回环地址的 HTTP 示例。

## 为什么这样改

前几个版本主要服务学习过程。v5 需要回答另一个问题：一个新用户如何知道安装什么、如何生成、服务监听在哪里、错误怎样返回。为此增加：

- `pyproject.toml`：声明包名、Python版本、零运行时第三方依赖和命令入口。
- `python -m ktorch.cli`：命令行生成入口，复用checkpoint完整性与词表校验。
- `python -m ktorch.server`：标准库HTTP示例，只监听 `127.0.0.1`，限制请求大小和生成长度。
- `examples/package_smoke.py` 与 `tests/test_package.py`：不需要模型文件的安装/导入检查。

## 目录

```text
pyproject.toml              setuptools构建元数据和console scripts
python/ktorch/cli.py        命令行生成
python/ktorch/server.py     本机HTTP示例
python/ktorch/__main__.py   python -m ktorch入口
examples/package_smoke.py   低负担包入口检查
tests/test_package.py       help与导入测试
design.md                   发布边界、错误处理、优化方向
```

运行不需要大模型：

```powershell
python examples/package_smoke.py
python -m unittest discover -s tests -v
python -m pip install --no-deps --no-build-isolation -e .
ktorch-reference-generate --help
ktorch-reference-serve --help
```

上面的可编辑安装需要本机已有 setuptools。真正的生成命令还需要先用同一版本训练出 checkpoint：

```powershell
python examples/train.py --steps 25
python -m ktorch.cli --checkpoint build/checkpoint.json --prompt 我 --tokens 8
python -m ktorch.server --checkpoint build/checkpoint.json --port 8765
```

## 支持边界

HTTP 示例只用于本机教学，不提供认证、TLS、限流、队列、监控、取消或多模型管理。它不能被称为企业级服务。v5 的工程目标是让安装、入口和错误边界可复现；生产服务需要另行设计。
