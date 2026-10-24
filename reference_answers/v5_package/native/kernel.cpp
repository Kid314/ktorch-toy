#include "core.hpp" // 使用统一的矩阵内核声明。
namespace kt { // 内核属于kt内部实现。
void gemm(const std::vector<double>& a,const std::vector<double>& b,std::vector<double>& c,int m,int k,int n){ // 计算C=A@B，C由调用者初始化为0。
    for(int i=0;i<m;++i){ // 遍历C的行。
        for(int j=0;j<n;++j){ // 遍历C的列。
            for(int p=0;p<k;++p){ // 沿乘法的共享维度累加。
                c[static_cast<std::size_t>(i)*n+j]+=a[static_cast<std::size_t>(i)*k+p]*b[static_cast<std::size_t>(p)*n+j]; // 一次乘加对应公式中的一个求和项。
            } // 当前C元素的求和结束。
        } // 当前输出行的列循环结束。
    } // 全部输出行计算结束。
} // 首版清晰的i-j-k内核结束。
} // kt命名空间结束。
