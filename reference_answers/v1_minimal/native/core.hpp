#pragma once // 防止同一个头文件被重复包含。
#include <cstdint> // 提供固定宽度的句柄整数类型。
#include <functional> // 保存每个算子的局部反向函数。
#include <memory> // 用shared_ptr表达计算图节点的共享身份。
#include <vector> // 用连续数组存储数值、梯度和输入节点。
namespace kt { // 内部C++接口统一放入kt命名空间。
struct Node; // 先声明节点，使它可以在别名中引用自己。
using Tensor = std::shared_ptr<Node>; // Tensor是同一个节点的共享句柄，复制句柄不复制数据。
struct Node { // 一个节点同时保存数值和生成这些数值的计算关系。
    int rows; // 二维Tensor的行数。
    int cols; // 二维Tensor的列数。
    std::vector<double> data; // 按行连续存储前向结果。
    std::vector<double> grad; // 保存损失对data每个元素的梯度。
    bool requires_grad; // 标记此节点是否参与自动求导。
    std::uint64_t version = 0; // 记录数据修改次数，防止反向使用过期的前向值。
    bool backward_used = false; // 首版每张图只允许执行一次backward。
    std::vector<Tensor> parents; // 结果持有输入，保证反向前输入仍然存活。
    std::vector<std::uint64_t> parent_versions; // 记录前向发生时输入节点的版本。
    std::function<void(const std::vector<double>&)> backward; // 接收上游梯度，不捕获自己的输出。
    Node(int r, int c, std::vector<double> values, bool track); // 构造时检查形状和数值。
    ~Node(); // 析构时更新诊断用的存活节点计数。
}; // Node定义结束。
bool grad_enabled(); // 查询当前线程是否记录计算图。
bool set_grad_enabled(bool enabled); // 设置记录开关并返回旧值，供no_grad恢复。
std::uint64_t live_nodes(); // 返回实际仍然存活的节点数，用于检查图释放。
Tensor make(int rows, int cols, std::vector<double> values, bool track=false); // 创建一个叶子Tensor。
Tensor result(int rows, int cols, std::vector<double> values, const std::vector<Tensor>& parents); // 创建算子结果并连接输入。
void backward(const Tensor& loss); // 从一个1×1损失开始反向遍历。
void replace(const Tensor& value, const std::vector<double>& data); // 只允许替换叶子数据并递增版本。
void zero_grad(const Tensor& value); // 清空该Tensor的梯度数组。
void sgd(const Tensor& value, double lr, double scale); // 用已算出的梯度执行一次SGD更新。
Tensor add(const Tensor& a,const Tensor& b); // 二维广播加法。
Tensor mul(const Tensor& a,const Tensor& b); // 二维广播逐元素乘法。
Tensor scale(const Tensor& a,double factor); // 乘一个不求导的常数。
Tensor matmul(const Tensor& a,const Tensor& b); // 二维矩阵乘法。
Tensor transpose(const Tensor& a); // 生成连续存储的转置副本。
Tensor gather(const Tensor& a,const std::vector<int>& ids); // 按整数索引查出若干行。
Tensor layer_norm(const Tensor& x,const Tensor& gamma,const Tensor& beta,double eps); // 每行独立的LayerNorm。
Tensor gelu(const Tensor& x); // 使用tanh近似的GELU。
Tensor softmax(const Tensor& x,bool causal); // 按行softmax，可限制为因果可见位置。
Tensor cross_entropy(const Tensor& x,const std::vector<int>& targets); // 从logits直接计算稳定的平均交叉熵。
void gemm(const std::vector<double>& a,const std::vector<double>& b,std::vector<double>& c,int m,int k,int n); // 矩阵计算内核。
} // kt命名空间结束。
