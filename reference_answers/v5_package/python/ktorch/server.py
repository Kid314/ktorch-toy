"""只用标准库的本机HTTP生成示例。"""  # 这是教学接口，不是生产服务。
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # 标准库提供并发请求线程。
from pathlib import Path  # 加载用户指定的checkpoint。
import argparse  # 配置监听地址和模型文件。
import json  # 请求和响应都使用JSON。
from .checkpoint import load  # 服务启动时加载一次模型。
from .training import generate  # 每个请求调用同一模型的推理路径。

def make_handler(model, vocab):  # 闭包绑定已加载模型，避免每请求重载文件。
    class Handler(BaseHTTPRequestHandler):  # 每个请求由标准库生成一个处理对象。
        def _json(self, status, payload):  # 统一写响应状态、类型和正文。
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")  # 中文保持可读。
            self.send_response(status)  # 写HTTP状态。
            self.send_header("Content-Type", "application/json; charset=utf-8")  # 声明编码。
            self.send_header("Content-Length", str(len(encoded)))  # 允许客户端正确读取正文。
            self.end_headers()  # 结束HTTP头。
            self.wfile.write(encoded)  # 写入完整响应。
        def do_GET(self):  # GET只提供一个健康检查。
            if self.path != "/health":  # 不暴露文件系统和任意路径。
                self._json(404, {"error": "use /health or POST /generate"})  # 给出接口提示。
                return  # 不继续处理未知路径。
            self._json(200, {"ok": True, "vocab_size": len(vocab)})  # 返回模型词表大小。
        def do_POST(self):  # 生成接口接受一个JSON对象。
            if self.path != "/generate":  # 当前服务只提供一个动作。
                self._json(404, {"error": "unknown path"})  # 不做隐式路由。
                return  # 结束本次请求。
            try:  # 解析、验证和生成都在同一错误边界内。
                length = int(self.headers.get("Content-Length", "0"))  # 读取客户端提供的正文长度。
                if length <= 0 or length > 1024 * 1024:  # 限制单次请求内存。
                    raise ValueError("request body size must be between 1 and 1MiB")  # 大请求需另设策略。
                payload = json.loads(self.rfile.read(length).decode("utf-8"))  # 只解析JSON，不执行代码。
                prompt = payload.get("prompt")  # prompt必须来自JSON字段。
                tokens = payload.get("tokens", 20)  # 默认短生成，避免本地服务长时间占用。
                if not isinstance(prompt, str) or not prompt or type(tokens) is not int or not 0 <= tokens <= 256:  # 限制输入形状和工作量。
                    raise ValueError("prompt must be nonempty and tokens must be 0..256")  # 明确错误原因。
                result = generate(model, vocab, prompt, tokens)  # 复用CLI的模型生成函数。
                self._json(200, {"text": result})  # 返回生成后的完整文本。
            except (UnicodeError, ValueError, TypeError) as exc:  # 用户请求错误返回400。
                self._json(400, {"error": str(exc)})  # 不泄露内部堆栈。
        def log_message(self, format, *args):  # 将标准服务器日志保持简短。
            print("server:", format % args)  # 保留状态码和路径便于手工观察。
    return Handler  # 返回绑定了模型的处理器类。

def main(argv=None):  # 启动本机服务的命令行入口。
    parser = argparse.ArgumentParser(description=__doc__)  # 生成服务帮助。
    parser.add_argument("--checkpoint", required=True)  # 明确模型文件来源。
    parser.add_argument("--host", default="127.0.0.1")  # 默认只监听本机回环地址。
    parser.add_argument("--port", type=int, default=8765)  # 教学默认端口。
    args = parser.parse_args(argv)  # 解析服务参数。
    if not 1 <= args.port <= 65535:  # 防止系统调用接收非法端口。
        parser.error("port must be between 1 and 65535")  # argparse以2退出。
    model, vocab, _ = load(Path(args.checkpoint))  # 服务启动前完成一次完整性检查。
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model, vocab))  # 绑定本机地址和处理器。
    print(f"listening on http://{args.host}:{args.port}")  # 显示实际监听位置。
    try:  # Ctrl+C是正常停止路径。
        server.serve_forever()  # 逐请求调用handler。
    except KeyboardInterrupt:  # 用户按Ctrl+C后进入清理。
        pass  # 不打印Python追踪栈。
    finally:  # 确保监听socket释放。
        server.server_close()  # 关闭HTTPServer。

if __name__ == "__main__":  # 支持python -m ktorch.server。
    main()  # 启动服务直到用户停止。
