#include "core.hpp" // 引入Tensor以及反向函数约定。
#include <algorithm> // 使用max和copy处理行数据。
#include <cmath> // 提供exp、log、sqrt和tanh。
#include <stdexcept> // 用异常报告形状与索引错误。
namespace kt { // 所有算子保持在同一内部命名空间。
namespace { // 私有辅助函数仅被本文件使用。
int dimension(int a,int b){if(a!=b&&a!=1&&b!=1)throw std::invalid_argument("broadcast mismatch");return std::max(a,b);} // 只有维度相等或其中一维为1才可广播。
std::size_t index(const Tensor& x,int i,int j){return static_cast<std::size_t>(x->rows==1?0:i)*x->cols+(x->cols==1?0:j);} // 广播维度始终读取第0项。
std::vector<double> transposed(const std::vector<double>& a,int rows,int cols){ // 转置普通数组，供前向和反向共用。
    std::vector<double> out(a.size()); // 分配与输入相同数量的元素。
    for(int i=0;i<rows;++i)for(int j=0;j<cols;++j)out[static_cast<std::size_t>(j)*rows+i]=a[static_cast<std::size_t>(i)*cols+j]; // 将(i,j)放到(j,i)。
    return out; // 返回连续存储的副本。
} // 数组转置结束。
void accumulate(const Tensor& x,const std::vector<double>& g){if(x->requires_grad)for(std::size_t i=0;i<g.size();++i)x->grad[i]+=g[i];} // 所有路径都累加到原参数身份。
} // 私有辅助区结束。
Tensor add(const Tensor& a,const Tensor& b){ // 实现二维广播加法。
    const int rows=dimension(a->rows,b->rows),cols=dimension(a->cols,b->cols); // 推导输出形状。
    std::vector<double> data(static_cast<std::size_t>(rows)*cols); // 分配输出数组。
    for(int i=0;i<rows;++i)for(int j=0;j<cols;++j)data[static_cast<std::size_t>(i)*cols+j]=a->data[index(a,i,j)]+b->data[index(b,i,j)]; // 每格相加。
    auto out=result(rows,cols,std::move(data),{a,b}); // 记录输入节点。
    if(out->requires_grad)out->backward=[a,b,rows,cols](const auto& g){ // 局部反向只捕获输入，不捕获out。
        for(int i=0;i<rows;++i)for(int j=0;j<cols;++j){ // 遍历上游输出梯度。
            const double value=g[static_cast<std::size_t>(i)*cols+j]; // 取当前输出格的梯度。
            if(a->requires_grad)a->grad[index(a,i,j)]+=value; // 广播索引重复自然完成求和归约。
            if(b->requires_grad)b->grad[index(b,i,j)]+=value; // 同一个输入参与多处时同样累加。
        } // 输出梯度遍历结束。
    }; // 加法局部反向定义结束。
    return out; // 返回结果节点。
} // add结束。
Tensor mul(const Tensor& a,const Tensor& b){ // 实现逐元素乘法，区别于矩阵乘法。
    const int rows=dimension(a->rows,b->rows),cols=dimension(a->cols,b->cols); // 验证并推导广播形状。
    std::vector<double> data(static_cast<std::size_t>(rows)*cols); // 分配结果。
    for(int i=0;i<rows;++i)for(int j=0;j<cols;++j)data[static_cast<std::size_t>(i)*cols+j]=a->data[index(a,i,j)]*b->data[index(b,i,j)]; // 对应元素相乘。
    auto out=result(rows,cols,std::move(data),{a,b}); // 连接计算图。
    if(out->requires_grad){ // 推理时不用保留反向缓存。
        const auto av=a->data,bv=b->data; // 保存本次前向参与计算的数值。
        out->backward=[a,b,av,bv,rows,cols](const auto& g){ // 对乘法两个输入分别求导。
            for(int i=0;i<rows;++i)for(int j=0;j<cols;++j){ // 逐格接收梯度。
                const auto ai=index(a,i,j),bi=index(b,i,j); // 找到广播前的实际输入格。
                const double value=g[static_cast<std::size_t>(i)*cols+j]; // 获取该格上游梯度。
                if(a->requires_grad)a->grad[ai]+=value*bv[bi]; // d(a*b)/da=b。
                if(b->requires_grad)b->grad[bi]+=value*av[ai]; // d(a*b)/db=a。
            } // 逐格反向结束。
        }; // 保存乘法反向函数。
    } // 梯度缓存分支结束。
    return out; // 返回乘积节点。
} // mul结束。
Tensor scale(const Tensor& a,double factor){ // 对全部元素乘常数。
    if(!std::isfinite(factor))throw std::invalid_argument("scale must be finite"); // 不接受非法缩放。
    auto data=a->data; // 创建结果副本，不原地覆盖输入。
    for(double& value:data)value*=factor; // 执行逐元素缩放。
    auto out=result(a->rows,a->cols,std::move(data),{a}); // 创建同形状结果。
    if(out->requires_grad)out->backward=[a,factor](const auto& g){if(a->requires_grad)for(std::size_t i=0;i<g.size();++i)a->grad[i]+=factor*g[i];}; // 反向也乘同一常数。
    return out; // 返回缩放结果。
} // scale结束。
Tensor matmul(const Tensor& a,const Tensor& b){ // 将矩阵乘法与局部反向组织在一起。
    if(a->cols!=b->rows)throw std::invalid_argument("matmul inner dimension mismatch"); // 检查共享维度。
    const int m=a->rows,k=a->cols,n=b->cols; // 明确A[m,k]和B[k,n]。
    std::vector<double> data(static_cast<std::size_t>(m)*n,0.0); // 内核使用累加，因此先清零。
    gemm(a->data,b->data,data,m,k,n); // 前向C=A@B。
    auto out=result(m,n,std::move(data),{a,b}); // 输出形状为[m,n]。
    if(out->requires_grad){ // 只在训练图中缓存输入。
        const auto av=a->data,bv=b->data; // 保留本次前向数值。
        out->backward=[a,b,av,bv,m,k,n](const auto& g){ // 接收G[m,n]。
            if(a->requires_grad){std::vector<double> ga(av.size(),0.0);gemm(g,transposed(bv,k,n),ga,m,n,k);accumulate(a,ga);} // dA=G@B.T，形状[m,k]。
            if(b->requires_grad){std::vector<double> gb(bv.size(),0.0);gemm(transposed(av,m,k),g,gb,k,m,n);accumulate(b,gb);} // dB=A.T@G，形状[k,n]。
        }; // 矩阵乘法反向定义结束。
    } // 训练缓存分支结束。
    return out; // 返回矩阵乘积。
} // matmul结束。
Tensor transpose(const Tensor& a){ // 第一版直接分配转置副本。
    auto out=result(a->cols,a->rows,transposed(a->data,a->rows,a->cols),{a}); // 交换行列。
    if(out->requires_grad)out->backward=[a](const auto& g){accumulate(a,transposed(g,a->cols,a->rows));}; // 梯度转置回原形状。
    return out; // 返回转置节点。
} // transpose结束。
Tensor gather(const Tensor& a,const std::vector<int>& ids){ // Embedding的本质是按行查表。
    if(ids.empty())throw std::invalid_argument("indices cannot be empty"); // 文本片段至少一个token。
    std::vector<double> data(ids.size()*a->cols); // 每个ID对应一行向量。
    for(std::size_t i=0;i<ids.size();++i){ // 遍历输入位置。
        if(ids[i]<0||ids[i]>=a->rows)throw std::out_of_range("embedding index out of range"); // 先检查词表边界。
        std::copy_n(a->data.begin()+static_cast<std::size_t>(ids[i])*a->cols,a->cols,data.begin()+i*a->cols); // 将指定行复制到输出位置。
    } // 查表结束。
    auto out=result(static_cast<int>(ids.size()),a->cols,std::move(data),{a}); // ID本身不是求导节点。
    if(out->requires_grad)out->backward=[a,ids](const auto& g){ // 反向将每个位置的梯度加回参数表。
        if(a->requires_grad)for(std::size_t i=0;i<ids.size();++i)for(int j=0;j<a->cols;++j)a->grad[static_cast<std::size_t>(ids[i])*a->cols+j]+=g[i*a->cols+j]; // 重复ID必须scatter-add。
    }; // 查表反向结束。
    return out; // 返回查到的向量矩阵。
} // gather结束。
Tensor layer_norm(const Tensor& x,const Tensor& gamma,const Tensor& beta,double eps){ // 对每一行独立归一化。
    if(gamma->rows!=1||beta->rows!=1||gamma->cols!=x->cols||beta->cols!=x->cols)throw std::invalid_argument("LayerNorm affine shape mismatch"); // gamma/beta必须是[1,d]。
    if(!std::isfinite(eps)||eps<=0)throw std::invalid_argument("epsilon must be positive"); // 保证方差分母有效。
    const int r=x->rows,d=x->cols; // r是token位置数，d是特征宽度。
    std::vector<double> normal(x->data.size()),inv(r),data(x->data.size()); // 保存标准化结果与反向需要的逆标准差。
    for(int i=0;i<r;++i){ // 各行之间不混合均值和方差。
        double mean=0.0,variance=0.0; // 初始化本行统计量。
        for(int j=0;j<d;++j)mean+=x->data[static_cast<std::size_t>(i)*d+j]; // 累加本行元素。
        mean/=d; // 得到总体均值。
        for(int j=0;j<d;++j){ // 对每个特征求相对本行均值的偏差。
            const double diff=x->data[static_cast<std::size_t>(i)*d+j]-mean; // 当前特征减去均值。
            variance+=diff*diff; // 将平方偏差加入总体方差的分子。
        } // 平方偏差累加结束。
        inv[i]=1.0/std::sqrt(variance/d+eps); // 总体方差除以d，不是d-1。
        for(int j=0;j<d;++j){ // 沿当前行产生输出。
            const auto k=static_cast<std::size_t>(i)*d+j; // 把二维位置映射到连续数组下标。
            normal[k]=(x->data[k]-mean)*inv[i]; // 标准化结果xhat=(x-mean)/sqrt(var+eps)。
            data[k]=normal[k]*gamma->data[j]+beta->data[j]; // 再用可训练的gamma/beta调整尺度与位置。
        } // 当前行的仿射输出结束。
    } // 全部行归一化结束。
    auto out=result(r,d,std::move(data),{x,gamma,beta}); // 三个输入都可能需要梯度。
    if(out->requires_grad){ // 推理时无需保存以下局部反向。
        const auto gv=gamma->data; // 固定本次前向的gamma。
        out->backward=[x,gamma,beta,normal,inv,gv,r,d](const auto& g){ // 同时求输入与两组仿射参数的梯度。
            for(int i=0;i<r;++i){ // 每行分别计算反向统计量。
                double total=0.0,weighted=0.0; // 对应Σq与Σ(q*xhat)。
                for(int j=0;j<d;++j){ // 先收集整行反向需要的公共项。
                    const auto k=static_cast<std::size_t>(i)*d+j; // 当前元素的连续下标。
                    const double q=g[k]*gv[j]; // q是损失对标准化结果xhat的梯度。
                    total+=q; // 累加Σq。
                    weighted+=q*normal[k]; // 累加Σ(q*xhat)。
                    if(gamma->requires_grad)gamma->grad[j]+=g[k]*normal[k]; // dgamma沿所有位置求和。
                    if(beta->requires_grad)beta->grad[j]+=g[k]; // dbeta同样沿所有位置求和。
                } // 公共项与仿射参数梯度已经就绪。
                if(x->requires_grad){ // 只有输入需要梯度时才计算dx。
                    for(int j=0;j<d;++j){ // 每个输入特征使用相同的行统计量。
                        const auto k=static_cast<std::size_t>(i)*d+j; // 定位当前输入特征。
                        x->grad[k]+=inv[i]/d*(d*g[k]*gv[j]-total-normal[k]*weighted); // 闭式导数同时包含均值和方差的依赖。
                    } // 输入特征的反向结束。
                } // 输入梯度分支结束。
            } // 每行反向结束。
        }; // LayerNorm反向定义结束。
    } // 训练分支结束。
    return out; // 返回与输入同形状的结果。
} // layer_norm结束。
Tensor gelu(const Tensor& x){ // 逐元素执行tanh近似GELU。
    constexpr double a=0.79788456080286535588; // sqrt(2/pi)的double常数。
    std::vector<double> data(x->data.size()),derivative(x->data.size()); // 输出及局部导数同形状。
    for(std::size_t i=0;i<data.size();++i){ // 每个元素独立变换。
        const double v=x->data[i],t=std::tanh(a*(v+0.044715*v*v*v)); // tanh内部包含三次项。
        data[i]=0.5*v*(1.0+t); // 前向f(x)=0.5x(1+tanh(u))。
        derivative[i]=0.5*(1.0+t)+0.5*v*(1.0-t*t)*a*(1.0+3.0*0.044715*v*v); // 对同一个近似公式求导。
    } // 元素变换结束。
    auto out=result(x->rows,x->cols,std::move(data),{x}); // 连接输入节点。
    if(out->requires_grad)out->backward=[x,derivative](const auto& g){if(x->requires_grad)for(std::size_t i=0;i<g.size();++i)x->grad[i]+=g[i]*derivative[i];}; // 链式法则乘局部导数。
    return out; // 返回非线性输出。
} // gelu结束。
Tensor softmax(const Tensor& x,bool causal){ // 对每行的可见位置计算稳定概率。
    if(causal&&x->rows!=x->cols)throw std::invalid_argument("causal attention scores must be square"); // 首版因果模式用于T×T注意力矩阵。
    const int r=x->rows,c=x->cols; // 保存矩阵尺寸。
    std::vector<double> p(x->data.size(),0.0); // 被遮住的位置从始至终保持0。
    for(int i=0;i<r;++i){ // 每行独立归一化。
        const int end=causal?i+1:c; // 因果模式保留对角线，只看j<=i。
        const auto base=static_cast<std::size_t>(i)*c; // 本行在连续数组中的起点。
        const double maximum=*std::max_element(x->data.begin()+base,x->data.begin()+base+end); // 只对可见位置找最大值。
        double total=0.0; // 指数分母从0开始。
        for(int j=0;j<end;++j){p[base+j]=std::exp(x->data[base+j]-maximum);total+=p[base+j];} // 减最大值避免溢出。
        for(int j=0;j<end;++j)p[base+j]/=total; // 得到可见位置总和为1的概率。
    } // 前向softmax结束。
    auto out=result(r,c,p,{x}); // 保存概率作为结果，也供局部反向使用。
    if(out->requires_grad)out->backward=[x,p,r,c,causal](const auto& g){ // 不显式创建完整Jacobian。
        if(!x->requires_grad)return; // 输入不求导时直接结束。
        for(int i=0;i<r;++i){ // 沿每行计算梯度。
            const int end=causal?i+1:c;const auto base=static_cast<std::size_t>(i)*c;double dot=0.0; // 先求Σ(g*p)。
            for(int j=0;j<end;++j)dot+=g[base+j]*p[base+j]; // 这是Jacobian乘向量中的公共项。
            for(int j=0;j<end;++j)x->grad[base+j]+=p[base+j]*(g[base+j]-dot); // dz=p*(g-Σgp)，未来梯度保持0。
        } // 每行梯度计算结束。
    }; // softmax反向定义结束。
    return out; // 返回概率节点。
} // softmax结束。
Tensor cross_entropy(const Tensor& x,const std::vector<int>& targets){ // 直接从logits求平均损失。
    if(targets.size()!=static_cast<std::size_t>(x->rows))throw std::invalid_argument("one target per row is required"); // 一个位置对应一个下一token标签。
    const int r=x->rows,c=x->cols;double loss=0.0;std::vector<double> p(x->data.size()); // 累积损失并保存概率。
    for(int i=0;i<r;++i){ // 分别处理每个训练位置。
        if(targets[i]<0||targets[i]>=c)throw std::out_of_range("target outside vocabulary"); // 防止标签越界。
        const auto base=static_cast<std::size_t>(i)*c; // 本行起点。
        const double maximum=*std::max_element(x->data.begin()+base,x->data.begin()+base+c); // 稳定log-sum-exp使用的偏移。
        double total=0.0; // 累积指数和。
        for(int j=0;j<c;++j){p[base+j]=std::exp(x->data[base+j]-maximum);total+=p[base+j];} // 指数不会因共同大偏移而溢出。
        for(int j=0;j<c;++j)p[base+j]/=total; // 反向使用概率。
        loss+=(maximum-x->data[base+targets[i]])+std::log(total); // 不先对可能下溢为0的目标概率取log。
    } // 所有位置的损失累积结束。
    auto out=result(1,1,{loss/r},{x}); // 训练目标对位置取平均。
    if(out->requires_grad)out->backward=[x,p,targets,r,c](const auto& g){ // 接收标量损失的上游梯度。
        if(x->requires_grad)for(int i=0;i<r;++i)for(int j=0;j<c;++j){const auto k=static_cast<std::size_t>(i)*c+j;x->grad[k]+=g[0]*(p[k]-(j==targets[i]?1.0:0.0))/r;} // dz=(p-onehot)/T，再乘上游梯度。
    }; // 交叉熵反向定义结束。
    return out; // 返回1×1标量Tensor。
} // cross_entropy结束。
} // kt命名空间结束。
