#include "core.hpp" // 引入节点和算子接口。
#include <algorithm> // 使用fill清空梯度。
#include <atomic> // 安全记录诊断用节点数量。
#include <cmath> // 使用isfinite检查非法数值。
#include <limits> // 防止形状乘法溢出。
#include <stdexcept> // 将输入错误转换成可读异常。
#include <unordered_set> // 拓扑遍历时按节点身份去重。
#include <utility> // 将数组所有权移动进节点。
namespace kt { // 开始实现内部Tensor功能。
namespace { // 仅本文件需要的状态不暴露给其他文件。
thread_local bool recording = true; // 每个线程拥有自己的no_grad状态。
std::atomic<std::uint64_t> count{0}; // 用于检验句柄和计算图是否及时释放。
} // 私有状态区结束。
Node::Node(int r,int c,std::vector<double> values,bool track) : rows(r),cols(c),data(std::move(values)),requires_grad(track) { // 保存参数并验证输入。
    if(r<=0||c<=0) throw std::invalid_argument("shape must be positive"); // 第一版不接受空Tensor。
    const auto size=static_cast<std::size_t>(r)*static_cast<std::size_t>(c); // 在无符号宽类型中计算元素个数。
    if(size!=data.size()||size>static_cast<std::size_t>(std::numeric_limits<int>::max())) throw std::invalid_argument("invalid tensor size"); // 检查数据长度和接口容量。
    for(double x:data) if(!std::isfinite(x)) throw std::invalid_argument("tensor data must be finite"); // 阻止NaN/Inf进入普通Tensor。
    grad.assign(size,0.0); // 梯度与数据同形状，初始为0。
    ++count; // 只有构造成功的节点才计入存活数量。
} // 节点构造结束。
Node::~Node(){--count;} // 最后一个共享引用离开后，记录节点已释放。
bool grad_enabled(){return recording;} // 返回当前线程的记录开关。
bool set_grad_enabled(bool enabled){const bool old=recording;recording=enabled;return old;} // 保存并切换开关。
std::uint64_t live_nodes(){return count.load();} // 读取诊断计数，不泄露指针。
Tensor make(int r,int c,std::vector<double> values,bool track){return std::make_shared<Node>(r,c,std::move(values),track);} // 创建新的叶子节点。
Tensor result(int r,int c,std::vector<double> values,const std::vector<Tensor>& inputs){ // 根据输入决定是否需要记录图。
    bool track=false; // 没有可训练输入时，结果默认不需要梯度。
    for(const auto& input:inputs) track=track||input->requires_grad; // 任一输入需要梯度即可传播需求。
    auto out=make(r,c,std::move(values),recording&&track); // no_grad可覆盖图记录需求。
    if(out->requires_grad){out->parents=inputs;for(const auto& input:inputs)out->parent_versions.push_back(input->version);} // 保存依赖及前向版本。
    return out; // 反向函数由具体算子补充。
} // 算子结果构造结束。
void zero_grad(const Tensor& value){std::fill(value->grad.begin(),value->grad.end(),0.0);} // 仅清当前节点的梯度。
void replace(const Tensor& value,const std::vector<double>& data){ // 提供参数加载和优化器的安全写入口。
    if(!value->parents.empty())throw std::invalid_argument("only leaf data may be replaced"); // 禁止覆盖中间激活。
    if(data.size()!=value->data.size())throw std::invalid_argument("replacement shape mismatch"); // 不允许通过写值改变形状。
    for(double x:data)if(!std::isfinite(x))throw std::invalid_argument("replacement must be finite"); // 先完整检查再写入。
    value->data=data; // 保存全部新值。
    ++value->version; // 让旧图能够发现参数已被改动。
} // 参数写入结束。
void backward(const Tensor& loss){ // 以标量损失作为反向的起点。
    if(loss->data.size()!=1||!loss->requires_grad)throw std::invalid_argument("backward needs a recorded scalar loss"); // 检查起点有效。
    if(loss->backward_used)throw std::runtime_error("run a fresh forward before backward again"); // 避免误用旧图重复累计。
    std::vector<Tensor> order; // 按输入在前、输出在后的顺序存节点。
    std::unordered_set<Node*> seen; // 同一个参数或激活只排序一次。
    std::function<void(const Tensor&)> visit=[&](const Tensor& node){ // 递归收集依赖图。
        if(!seen.insert(node.get()).second)return; // 共享节点不重复展开。
        for(const auto& parent:node->parents)visit(parent); // 先访问输入，建立拓扑顺序。
        order.push_back(node); // 输入已加入后，再加入当前节点。
    }; // 拓扑收集器定义结束。
    visit(loss); // 找到本次损失实际依赖的全部节点。
    for(const auto& node:order){ // 在改梯度之前统一验证图仍然有效。
        for(std::size_t i=0;i<node->parents.size();++i)if(node->parents[i]->version!=node->parent_versions[i])throw std::runtime_error("input modified after forward"); // 拒绝过期前向图。
        if(!node->parents.empty())zero_grad(node); // 中间梯度从0开始，叶子梯度按约定允许累计。
    } // 图验证与中间梯度清理结束。
    loss->grad[0]+=1.0; // dL/dL为1；叶子标量也遵守累加语义。
    for(auto it=order.rbegin();it!=order.rend();++it)if((*it)->backward)(*it)->backward((*it)->grad); // 逆拓扑逐个执行局部反向。
    loss->backward_used=true; // 标记这张损失图已经使用过。
} // 完整反向执行结束。
void sgd(const Tensor& value,double lr,double factor){ // 只更新登记为可训练叶子的参数。
    if(!value->requires_grad||!value->parents.empty())throw std::invalid_argument("optimizer expects trainable leaf"); // 防止更新激活。
    if(!std::isfinite(lr)||lr<0||!std::isfinite(factor)||factor<0)throw std::invalid_argument("invalid learning rate or gradient scale"); // 检查超参数。
    auto next=value->data; // 先计算候选值，发生错误时不留下半个参数更新。
    for(std::size_t i=0;i<next.size();++i)next[i]-=lr*factor*value->grad[i]; // theta_new=theta-lr*gradient。
    replace(value,next); // 统一验证有限值并递增参数版本。
} // SGD结束。
} // kt命名空间结束。
