角色： 你是一名精通科学计算和数值优化的 Python 专家，擅长使用 SciPy、NumPy 等库解决复杂的非线性优化问题。

任务： 请为我编写一个 Python 脚本，用于求解一个非线性约束优化问题。该问题旨在为一个给定的损失预测模型（Loss Prediction Model）拟合最佳参数。你将使用 scipy.optimize.minimize 来实现这一目标。

问题背景与数学模型：

我们进行了一系列实验，通过改变 code 数据集的数量，得到了模型在总验证集上的损失（validation loss）。现在，我们需要根据这些实验数据，拟合一个模型的参数。

损失预测模型 (L_tilde)：
预测的损失 L_tilde 由以下公式给出：

$$\\tilde{L}(N\_{code}^{(t)}) = C \\cdot (N\_{code}^{(t)} + k \\cdot |N - N\_{code}^{(0)}|^{\\alpha})^{-\\beta} + E $$
其中：

N_code 
(t)
  是在第 t 次实验中 code 数据集的 token 数量。

∣N−N_code 
(0)
 ∣ 是其他领域（math + science）的总 token 数量，这是一个常数。

C,k,
alpha,
beta,E 是我们要优化的五个参数。

目标函数 (Objective Function)：
我们的目标是最小化 预测损失 与 真实观测损失 之间的残差。我们使用 Huber Loss 来度量这个残差，并对所有实验点求和。

$$\\min\_{C, k, \\alpha, \\beta, E} \\sum\_{t=0}^{4} \\text{Huber}*{\\delta} (\\tilde{L}(N*{code}^{(t)}) - L\_{code}^{(t)}) $$
其中：

L_code 
(t)
  是第 t 次实验中观测到的真实总验证集 loss。

Huber Loss 的参数 
delta=0.001。

非线性约束 (Nonlinear Constraint)：
参数必须满足以下约束，以确保从其他领域转移的“有效数据量”不超过这些领域的原始数据量：

$$k \\cdot |N - N\_{code}|^{\\alpha} \\le |N - N\_{code}| $$

实验数据和常量：

我已经为你预处理好了实验数据。

基础数据量：

code 数据集的基础 token 数 N_code 
(0)
 =660,000。

其他领域数据（math+science）的总 token 数 ∣N−N_code∣=660,000(
textmath)+660,000(
textscience)=1,320,000。

实验数据点 (t=0 to 4):
我们有5个实验数据点，包括基线和对 code 数据集的4种扰动。

N_code 
(t)
 : code 数据集的 token 数量。

L_code 
(t)
 : 对应的真实 loss (V1+V2+V3 的总和)。

Python

# 实验数据
# 格式: (N_code_t, L_code_t_observed)
experimental_data = [
    # t=0: baseline (1x code)
    (660000 * 1, 0.5655004955709927 + 0.7627777529485298 + 1.1549238205459047),
    # t=1: code_1_3 (1/3x code)
    (660000 * (1/3), 0.5609681899585421 + 0.8135941916643971 + 1.1464412223689164),
    # t=2: code_1_2 (1/2x code)
    (660000 * (1/2), 0.5338599962137994 + 0.7633443399511203 + 1.0913624213969155),
    # t=3: code_2x (2x code)
    (660000 * 2, 0.5811840244938457 + 0.7418261641504789 + 1.182430182328204),
    # t=4: code_3x (3x code)
    (660000 * 3, 0.5067948727380662 + 0.5935674406061269 + 1.0421503492422748)
]
代码实现要求：

库： 请使用 numpy 和 scipy.optimize.minimize。

优化方法： 请使用 trust-constr 方法，因为它适用于处理非线性约束，与论文中提到的 "Trust Region Method" 一致。

脚本结构：

在脚本开头定义所有常量和上述 experimental_data。

定义一个函数 predict_loss(params, N_code_t)，根据公式计算 
tildeL。params 是一个包含 [C, k, alpha, beta, E] 的 NumPy 数组。

定义 huber_loss(residual, delta) 函数。

定义 目标函数 objective_func(params)，该函数应返回所有数据点的 Huber Loss 之和。

定义 约束函数 constraint_func(params)，并将其构造成 scipy 所需的格式（例如，NonlinearConstraint 对象）。约束应为 g(x)
le0 的形式，即 k
cdot∣N−N_code∣ 
alpha
 −∣N−N_code∣
le0。

参数设置：

初始猜测 (x0)： 为 [C, k, alpha, beta, E] 提供一个合理的初始猜测值，例如 [1.0, 1.0, 1.0, 0.5, 0.1]。

参数边界 (bounds)： 为参数设置边界。通常 C, k, beta, E 应为正数。例如，C > 0, k > 0, beta > 0。可以为 alpha 设置一个较宽的范围，例如 [-5, 5]。

输出：

调用 scipy.optimize.minimize 执行优化。

清晰地打印出优化结果，包括：

优化是否成功。

最终的目标函数值（总 Huber Loss）。

找到的最佳参数值 C, k, alpha, beta, E。

请确保代码是自包含的、可直接运行的，并添加适当的注释以解释关键步骤。