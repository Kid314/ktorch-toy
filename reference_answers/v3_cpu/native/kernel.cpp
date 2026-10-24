#include "core.hpp" // 使用统一的矩阵内核声明。
#include <cstdlib> // 读取KTORCH_GEMM环境变量。
#include <string> // 保存可读的策略名称。
namespace kt { // 内核属于kt内部实现。
void gemm(const std::vector<double>& a,const std::vector<double>& b,std::vector<double>& c,int m,int k,int n){ // 计算C=A@B，C由调用者初始化为0。
    const char* selected=std::getenv("KTORCH_GEMM"); // 每次调用读取环境，方便同一DLL比较不同策略。
    const std::string strategy=selected?selected:"ijk"; // 默认使用最直观的i-j-k。
    if(strategy=="ikj"){ // i-k-j让A元素复用，并顺序写入C的一行。
        for(int i=0;i<m;++i){ // 遍历输出行。
            for(int p=0;p<k;++p){ // 先固定共享维度元素。
                const double left=a[static_cast<std::size_t>(i)*k+p]; // 只读取一次A[i,p]。
                for(int j=0;j<n;++j){ // 连续更新C的第i行。
                    c[static_cast<std::size_t>(i)*n+j]+=left*b[static_cast<std::size_t>(p)*n+j]; // 仍然是同一个乘加公式。
                } // 当前A[i,p]对整行的贡献结束。
            } // 当前输出行的共享维度结束。
        } // ikj策略结束。
        return; // 不再执行ijk分支。
    } // 策略选择结束。
    for(int i=0;i<m;++i){ // 朴素ijk按输出格逐个计算，便于对照公式。
        for(int j=0;j<n;++j){ // 遍历C的列。
            for(int p=0;p<k;++p){ // 沿乘法的共享维度累加。
                c[static_cast<std::size_t>(i)*n+j]+=a[static_cast<std::size_t>(i)*k+p]*b[static_cast<std::size_t>(p)*n+j]; // 一次乘加对应公式中的一个求和项。
            } // 当前C元素的求和结束。
        } // 当前输出行的列循环结束。
    } // 全部输出行计算结束。
} // 首版清晰的i-j-k内核结束。
} // kt命名空间结束。
