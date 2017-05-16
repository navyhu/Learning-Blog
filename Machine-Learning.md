# Machine Learning (Andrew Ng)

## Introduction
### Definition
对于一个计算机程序，存在任务T，以及性能测量方法P，如果在经验E的影响下，通过P的测量，程序完成任务T的性能提高了，就说此程序从经验E学习到了。
- 监督学习(Supervised Leanring)
  * 房价预测 回归问题（regression problem），预测数据是连续的
  * 恶性肿瘤预测 分类问题，非连续数据预测
- 学习理论(Learning Theory)
- 非监督学习(Unsupervised Learning)
  * 聚类问题，提供一组数据给学习算法，找出里面的模式，进行分组
  * 可以把图片分成不同区域
- 强化学习(Reinforcement Learning)
  * 回报函数，跟训练狗相似，表现好就正反馈，表现不好就负反馈

## Model representation
x(i) :第i个输入变量

## Linear Regression
### Linear Regression with one variable
- Hypothesis 预测函数
  - hθ(x) = θ0 + θ1 \* x;
  - hθ(x) = θ\*X; (矩阵形式)

- Cost Function 代价函数
  - J(θ0, θ1) = (1/2m) \* sum(i = 1, m)[(hθ(x(i)) - y(i))^2];
  - m为训练集(X)数量

- Goal
  - 选取θ0, θ1, 使J(θ0, θ1)最小

#### Gradient Descent 梯度下降获取θ
- 步骤
  - 选取初始θ值
  - 改变θ值使J(θ)变小，直到取到J(θ)最小值


