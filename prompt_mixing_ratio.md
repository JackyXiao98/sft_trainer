角色: 你是一位顶尖的运筹学和数值优化专家，擅长使用 Python 的 SciPy 库来解决带约束的非线性规划问题。

任务: 基于先前计算出的一组 scaling law 参数，你需要编写一个 Python 脚本来求解一个数据配比的优化问题。目标是在给定的总数据预算 (N_0)下，找到三个数据集（Math, Code, Science）的最佳混合权重 (w_i)，以最小化预测的总损失。

问题背景与数学模型：

我们已经为 Math, Code, Science (对应论文中的 IF) 三个领域分别拟合了 scaling law 参数 (C_i,k_i,
alpha_i,
beta_i,E_i)。现在，我们需要根据以下优化模型来分配总计 N_0 的数据预算：

目标函数 (Objective Function):
最小化三个领域预测损失的总和。权重向量 w=[w_math,w_code,w_science] 是我们的优化变量。

$$\\min\_{\\mathbf{w}} \\sum\_{i=1}^{K=3} \\left[ C\_i \\cdot (w\_i N\_0 + k\_i (N\_0 - w\_i N\_0)^{\\alpha\_i})^{-\\beta\_i} + E\_i \\right] $$

约束条件 (Constraints):

权重边界 (Bounds): 每个领域的权重必须在 0 和 1 之间。

$$0 \\le w\_i \\le 1 \\quad \\forall i \\in {1, 2, 3}$$

等式约束 (Equality Constraint): 所有领域的权重之和必须等于 1。

$$\\sum\_{i=1}^{K=3} w\_i = 1$$

数据和常量：

总数据预算: N_0=120,000,000 (120 Million tokens)。

领域数量: K=3。

Scaling Law 参数 (来自图1): 我已经将参数整理成一个 Python 字典。请注意，Science 领域对应于图中的 IF (Instruction Following) 领域。

Python

# Pre-calculated scaling law parameters for each domain
# Note: 'science' domain corresponds to 'IF' in the table.
parameters = {
    "math": {
        "C": 0.7512, "k": 0.0401, "alpha": 0.4467, "beta": 0.0430, "E": 1.4934
    },
    "code": {
        "C": 0.9820, "k": 0.1235, "alpha": 0.5235, "beta": 0.0439, "E": 1.2679
    },
    "science": { # Mapped from 'IF'
        "C": 1.1562, "k": 0.1948, "alpha": 0.5288, "beta": 0.0510, "E": 1.0967
    }
}
# Ordered list of domains to match the weight vector w = [w_math, w_code, w_science]
domains = ["math", "code", "science"]
代码实现要求：

库: 请使用 numpy 和 scipy.optimize.minimize。

优化算法: 严格按照论文中的建议，使用 SLSQP (Sequential Least Squares Programming) 算法，因为它非常适合处理带边界和等式约束的非线性优化问题。

脚本结构:

在脚本顶部定义常量 N_0 和 parameters 字典。

定义一个 目标函数 objective_function(w)，它接收一个权重向量 w (NumPy array) 作为输入，并返回所有领域的预测损失之和。函数内部应按 domains 列表的顺序迭代，从 parameters 字典中提取相应参数进行计算。

定义 约束条件:

使用 Bounds 对象来处理 0
lew_i
le1 的边界约束。

使用一个字典来定义 
sumw_i=1 的等式约束，格式为 {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}。

参数设置:

初始猜测 (x0): 使用一个均匀分布的权重作为起点，即 [1/3, 1/3, 1/3]。

输出:

调用 scipy.optimize.minimize 并传入目标函数、初始猜测、方法(SLSQP)、边界和约束。

清晰地打印最终的优化结果：

优化过程是否成功 (result.success)。

最终的最小化总损失 (result.fun)。

为 Math, Code, Science 计算出的最优权重，最好同时显示为小数和百分比。

打印最终权重的总和，以验证其是否为 1。

请确保代码逻辑清晰，有适当的注释，并且可以直接运行。