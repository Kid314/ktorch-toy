"""python -m ktorch 等价于v5的生成CLI。"""  # 包本身不自动启动服务。
from .cli import main  # 复用同一个参数和错误处理。
raise SystemExit(main())  # 让模块入口与console_script一致。
