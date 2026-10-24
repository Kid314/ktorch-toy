#include "core.hpp" // 复用Tensor和安全写入接口。
#include <cmath> // 提供幂、平方根和有限值检查。
#include <stdexcept> // 错误在C ABI边界转换。
namespace kt { // AdamW属于原生数值更新。
void adamw(const Tensor& p,const Tensor& m,const Tensor& v,int step,double lr,double b1,double b2,double eps,double decay,double factor){ // Python只调度，更新计算在C++。
    if(!p->requires_grad||!p->parents.empty()||m->requires_grad||v->requires_grad)throw std::invalid_argument("AdamW requires leaf parameter and constant moments"); // 矩状态不是模型梯度参数。
    if(p==m||p==v||m==v||m->rows!=p->rows||m->cols!=p->cols||v->rows!=p->rows||v->cols!=p->cols)throw std::invalid_argument("AdamW state identity/shape mismatch"); // 三份数值需要独立且同形状。
    if(step<=0||!std::isfinite(lr)||lr<=0||!(b1>=0&&b1<1)||!(b2>=0&&b2<1)||!std::isfinite(eps)||eps<=0||!std::isfinite(decay)||decay<0||!std::isfinite(factor)||factor<0)throw std::invalid_argument("invalid AdamW hyperparameters"); // NaN也不能通过范围条件。
    auto next_p=p->data,next_m=m->data,next_v=v->data; // 先算全部候选值，检查后统一写回。
    const double correction1=1.0-std::pow(b1,step); // 修正一阶矩从0初始化带来的偏差。
    const double correction2=1.0-std::pow(b2,step); // 修正二阶矩的初始化偏差。
    for(std::size_t i=0;i<next_p.size();++i){ // 每个参数元素有对应的m和v。
        const double g=p->grad[i]*factor; // 使用全局裁剪之后的梯度。
        next_m[i]=b1*m->data[i]+(1.0-b1)*g; // m_t=beta1*m_(t-1)+(1-beta1)*g。
        next_v[i]=b2*v->data[i]+(1.0-b2)*g*g; // v_t=beta2*v_(t-1)+(1-beta2)*g²。
        const double normalized=(next_m[i]/correction1)/(std::sqrt(next_v[i]/correction2)+eps); // 使用偏差修正后的自适应更新。
        next_p[i]=p->data[i]*(1.0-lr*decay)-lr*normalized; // AdamW衰减与梯度更新解耦。
        if(!std::isfinite(next_p[i])||!std::isfinite(next_m[i])||!std::isfinite(next_v[i])||next_v[i]<0)throw std::runtime_error("nonfinite AdamW update"); // 本参数的任何非法值都阻止写入。
    } // 全部候选元素已检查。
    replace(p,next_p); // 写回模型参数并递增数据版本。
    replace(m,next_m); // 保存下一轮的一阶矩。
    replace(v,next_v); // 保存下一轮的二阶矩。
} // AdamW数值更新结束。
} // kt命名空间结束。
