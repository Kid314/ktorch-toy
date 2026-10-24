"""用本机C++17编译器构建ctypes动态库；编译过程串行执行。"""  # 这个脚本不训练模型。
from pathlib import Path  # 从脚本自身位置定位源码，避免依赖当前终端目录。
import argparse  # 支持用户显式传入编译器路径。
import os  # 为编译子进程准备运行库查找路径。
import shutil  # 查找PATH中的编译器并复制成功产物。
import subprocess  # 以参数列表调用编译器，不拼接shell命令。
import sys  # 判断平台和输出退出码。

ROOT = Path(__file__).resolve().parents[1]  # 本版目录是scripts的父目录。

def find_compiler(explicit=None):  # 优先使用用户指定的工具链。
    if explicit:  # 命令行选择应覆盖自动探测。
        candidate = shutil.which(explicit) or explicit  # 同时支持名字和完整路径。
        if not Path(candidate).is_file():  # 提前给出可理解的缺失提示。
            raise RuntimeError(f"编译器不存在：{explicit}")  # 不继续生成无效编译命令。
        return str(candidate)  # 返回已验证存在的路径。
    if os.environ.get("CXX"):  # 允许当前终端通过环境变量指定编译器。
        return find_compiler(os.environ["CXX"])  # 复用相同的路径验证。
    if os.name == "nt":  # Windows上优先使用带标准库的完整MinGW工具链。
        base = Path(os.environ.get("ProgramFiles", "C:/Program Files"))  # 读取安装目录而不改系统变量。
        found = sorted(base.glob("JetBrains/CLion*/bin/mingw/bin/g++.exe"), reverse=True)  # 探测CLion附带的编译器。
        if found:  # 本机已有工具链时直接使用。
            return str(found[0])  # 取一个确定的候选。
    for name in ("g++", "clang++", "c++"):  # 其他平台使用常见编译器名字。
        candidate = shutil.which(name)  # 仅查找当前PATH。
        if candidate:  # 找到后由真实编译验证其可用性。
            return candidate  # 返回完整可执行路径。
    raise RuntimeError("没有找到C++17编译器，请使用 --compiler 指定g++或clang++。")  # 缺工具时给出下一步。

def build(explicit=None):  # 构建函数也可被打包脚本调用。
    compiler = find_compiler(explicit)  # 确定本次真实使用的编译器。
    suffix = ".dll" if os.name == "nt" else ".dylib" if sys.platform == "darwin" else ".so"  # 选择系统动态库扩展名。
    folder = ROOT / "build"  # 临时产物统一放进本版build目录。
    folder.mkdir(exist_ok=True)  # 不要求用户手动创建构建目录。
    staged = folder / ("_native" + suffix)  # 先编译临时目标，失败不破坏已可用库。
    target = ROOT / "python" / "ktorch" / staged.name  # Python从自己的包目录加载动态库。
    flags = ["-std=c++17", "-O2", "-fno-fast-math"]  # 保留浮点语义，不用激进近似牺牲正确性。
    flags += ["-shared", "-static", "-static-libgcc", "-static-libstdc++"] if os.name == "nt" else ["-fPIC", "-dynamiclib" if sys.platform == "darwin" else "-shared"]  # Windows将GCC运行库静态带入。
    sources = sorted((ROOT / "native").glob("*.cpp"))  # 每版新增的原生源文件自动纳入构建。
    command = [compiler, *flags, "-I", str(ROOT / "native"), *map(str, sources), "-o", str(staged)]  # 文件名有空格也作为单独参数传入。
    env = os.environ.copy()  # 不修改父进程或系统的PATH。
    env["PATH"] = str(Path(compiler).parent) + os.pathsep + env.get("PATH", "")  # 编译子进程能找到cc1plus和工具链运行库。
    print("编译器：", compiler, flush=True)  # 保留可复核的工具链信息。
    subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=180)  # 等待一次串行编译，失败立即报错。
    target.parent.mkdir(parents=True, exist_ok=True)  # 确保安装到包目录的路径存在。
    shutil.copy2(staged, target)  # 只有成功构建才替换Python要加载的库。
    print("已构建：", target)  # 给出实际产物位置。
    return target  # 打包脚本可以继续使用这份产物。

if __name__ == "__main__":  # 导入本脚本不会自动编译。
    parser = argparse.ArgumentParser(description=__doc__)  # 生成简短帮助。
    parser.add_argument("--compiler", help="g++/clang++的完整路径")  # 提供手动覆盖入口。
    args = parser.parse_args()  # 读取当前命令行。
    try:  # 将常见构建失败转成清晰退出信息。
        build(args.compiler)  # 进行实际编译。
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:  # 包括缺工具、链接失败和文件被占用。
        raise SystemExit(f"构建失败：{exc}") from exc  # 非零退出，不能误称构建成功。
