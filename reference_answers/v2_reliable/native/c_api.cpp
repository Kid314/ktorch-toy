#include "core.hpp" // 使用C++内部实现，外部只看到C兼容函数。
#include <algorithm> // 用copy将数值复制到Python提供的数组。
#include <mutex> // 保护全局句柄表，避免多个ctypes调用破坏容器。
#include <stdexcept> // 报告无效句柄和参数。
#include <string> // 保存线程自己的错误消息。
#include <unordered_map> // 将整数句柄映射到共享节点。
#ifdef _WIN32 // Windows动态库需要显式导出符号。
#define API extern "C" __declspec(dllexport) // 关闭C++名字改编并导出C函数。
#else // Unix共享库使用默认可见符号。
#define API extern "C" __attribute__((visibility("default"))) // 统一两种平台的导出写法。
#endif // 平台导出宏定义结束。
namespace { // 这些绑定状态不属于对外ABI。
using H=std::uint64_t; // ABI句柄固定为64位无符号整数，不暴露原始C++指针。
std::unordered_map<H,kt::Tensor> handles; // 表中每项持有一个共享节点引用。
H next_handle=1; // 0保留为失败返回值，已释放ID不复用。
std::mutex mutex; // 每次ABI调用在一个临界区内解析句柄并完成运算。
thread_local std::string error; // 不同调用线程拥有独立错误消息。
kt::Tensor get(H h){const auto it=handles.find(h);if(it==handles.end())throw std::invalid_argument("invalid or released handle");return it->second;} // 先验证ID再返回节点。
H keep(kt::Tensor value){const H h=next_handle++;handles.emplace(h,std::move(value));return h;} // 为返回结果创建一个Python需要释放的句柄。
template<class F> H tensor_call(F f)noexcept{try{std::lock_guard<std::mutex> lock(mutex);error.clear();return keep(f());}catch(const std::exception& e){error=e.what();return 0;}catch(...){error="unknown native exception";return 0;}} // 异常不能穿过C ABI。
template<class F> int status_call(F f)noexcept{try{std::lock_guard<std::mutex> lock(mutex);error.clear();f();return 0;}catch(const std::exception& e){error=e.what();return -1;}catch(...){error="unknown native exception";return -1;}} // 状态接口以0表示成功。
std::vector<int> ids(const int* p,int n){if(!p||n<=0)throw std::invalid_argument("indices must be nonempty");return {p,p+n};} // 根据显式长度复制整数数组。
} // 绑定辅助区结束。
API int kt_abi_version()noexcept{return 1;} // Python可先核对ABI版本。
API const char* kt_last_error()noexcept{return error.c_str();} // Python必须及时复制该消息。
API H kt_create(const double* p,int r,int c,int track)noexcept{return tensor_call([&]{if(!p||r<=0||c<=0||static_cast<std::uint64_t>(r)*c>10000000)throw std::invalid_argument("invalid or oversized shape");return kt::make(r,c,{p,p+static_cast<std::size_t>(r)*c},track!=0);});} // 创建时复制输入，避免借用Python临时数组。
API int kt_release(H h)noexcept{return status_call([&]{if(handles.erase(h)!=1)throw std::invalid_argument("handle already released");});} // 删除一个句柄引用，图中的其他引用仍有效。
API int kt_shape(H h,int* out)noexcept{return status_call([&]{if(!out)throw std::invalid_argument("null shape buffer");const auto x=get(h);out[0]=x->rows;out[1]=x->cols;});} // 输出两个整数维度。
API int kt_data(H h,double* out,int n,int gradient)noexcept{return status_call([&]{const auto x=get(h);if(!out||n!=static_cast<int>(x->data.size()))throw std::invalid_argument("copy size mismatch");const auto& data=gradient?x->grad:x->data;std::copy(data.begin(),data.end(),out);});} // 统一复制数值或梯度，不暴露悬空指针。
API int kt_replace(H h,const double* p,int n)noexcept{return status_call([&]{const auto x=get(h);if(!p||n!=static_cast<int>(x->data.size()))throw std::invalid_argument("replace size mismatch");kt::replace(x,{p,p+n});});} // 用安全入口加载参数。
API int kt_requires_grad(H h)noexcept{int answer=-1;status_call([&]{answer=get(h)->requires_grad?1:0;});return answer;} // -1表示句柄错误。
API H kt_add(H a,H b)noexcept{return tensor_call([&]{return kt::add(get(a),get(b));});} // 暴露广播加法。
API H kt_mul(H a,H b)noexcept{return tensor_call([&]{return kt::mul(get(a),get(b));});} // 暴露逐元素乘法。
API H kt_scale(H a,double c)noexcept{return tensor_call([&]{return kt::scale(get(a),c);});} // 暴露常数缩放。
API H kt_matmul(H a,H b)noexcept{return tensor_call([&]{return kt::matmul(get(a),get(b));});} // 暴露矩阵乘法。
API H kt_transpose(H a)noexcept{return tensor_call([&]{return kt::transpose(get(a));});} // 暴露二维转置。
API H kt_gather(H a,const int* p,int n)noexcept{return tensor_call([&]{return kt::gather(get(a),ids(p,n));});} // 暴露按行查表。
API H kt_layer_norm(H x,H gamma,H beta,double eps)noexcept{return tensor_call([&]{return kt::layer_norm(get(x),get(gamma),get(beta),eps);});} // 三个Tensor参数均经句柄表验证。
API H kt_gelu(H x)noexcept{return tensor_call([&]{return kt::gelu(get(x));});} // 暴露非线性运算。
API H kt_softmax(H x,int causal)noexcept{return tensor_call([&]{return kt::softmax(get(x),causal!=0);});} // causal控制是否使用下三角可见域。
API H kt_cross_entropy(H x,const int* p,int n)noexcept{return tensor_call([&]{return kt::cross_entropy(get(x),ids(p,n));});} // 标签被复制并校验。
API int kt_backward(H loss)noexcept{return status_call([&]{kt::backward(get(loss));});} // 反向失败返回可读错误。
API int kt_zero_grad(H x)noexcept{return status_call([&]{kt::zero_grad(get(x));});} // 清参数梯度。
API int kt_sgd(H x,double lr,double factor)noexcept{return status_call([&]{kt::sgd(get(x),lr,factor);});} // 原生数值更新不建立新图。
API int kt_grad_mode(int enabled)noexcept{return kt::set_grad_enabled(enabled!=0)?1:0;} // 返回原开关以支持嵌套no_grad。
API std::uint64_t kt_live_nodes()noexcept{return kt::live_nodes();} // 用于小规模生命周期检查。
API int kt_adamw(H p,H m,H v,int step,double lr,double b1,double b2,double eps,double decay,double factor)noexcept{return status_call([&]{kt::adamw(get(p),get(m),get(v),step,lr,b1,b2,eps,decay,factor);});} // v2扩展的原生优化器接口。
