#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据配比优化脚本

该脚本基于预先计算的scaling law参数，求解在给定总数据预算下
三个领域（Math, Code, Science）的最佳混合权重，以最小化预测的总损失。

使用SLSQP (Sequential Least Squares Programming) 算法进行优化。

作者: AI Assistant
日期: 2024
"""

import numpy as np
from scipy.optimize import minimize, Bounds
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 常量和参数定义
# ============================================================================

# 总数据预算 (120 Million tokens)
N_0 = 120_000_000

# 预先计算的scaling law参数
# 注意: 'science'领域对应论文中的'IF' (Instruction Following)领域
parameters = {
    "math": {
        "C": 0.7512, "k": 0.0401, "alpha": 0.4467, "beta": 0.0430, "E": 1.4934
    },
    "code": {
        "C": 0.9820, "k": 0.1235, "alpha": 0.5235, "beta": 0.0439, "E": 1.2679
    },
    "science": {  # 映射自'IF'
        "C": 1.1562, "k": 0.1948, "alpha": 0.5288, "beta": 0.0510, "E": 1.0967
    }
}

# 领域顺序列表，与权重向量 w = [w_math, w_code, w_science] 对应
domains = ["math", "code", "science"]

print("数据配比优化问题")
print("=" * 50)
print(f"总数据预算: {N_0:,} tokens ({N_0/1_000_000:.0f}M tokens)")
print(f"领域数量: {len(domains)}")
print(f"领域: {', '.join(domains)}")
print()

print("Scaling Law 参数:")
for domain in domains:
    params = parameters[domain]
    print(f"  {domain.capitalize()}:")
    print(f"    C={params['C']:.4f}, k={params['k']:.4f}, α={params['alpha']:.4f}")
    print(f"    β={params['beta']:.4f}, E={params['E']:.4f}")
print()

# ============================================================================
# 目标函数定义
# ============================================================================

def objective_function(w):
    """
    目标函数: 计算所有领域预测损失的总和
    
    公式: min_w Σ[C_i * (w_i*N_0 + k_i*(N_0 - w_i*N_0)^α_i)^(-β_i) + E_i]
    
    参数:
        w: 权重向量 [w_math, w_code, w_science]
    
    返回:
        总预测损失
    """
    total_loss = 0.0
    
    # 按domains列表顺序迭代，从parameters字典中提取相应参数
    for i, domain in enumerate(domains):
        # 获取当前领域的参数
        params = parameters[domain]
        C_i = params["C"]
        k_i = params["k"]
        alpha_i = params["alpha"]
        beta_i = params["beta"]
        E_i = params["E"]
        
        # 获取当前领域的权重
        w_i = w[i]
        
        # 计算当前领域的数据量
        domain_data = w_i * N_0
        other_data = N_0 - w_i * N_0
        
        # 计算有效数据量: w_i*N_0 + k_i*(N_0 - w_i*N_0)^α_i
        effective_data = domain_data + k_i * (other_data ** alpha_i)
        
        # 避免数值问题
        if effective_data <= 0:
            return float('inf')
        
        # 计算当前领域的预测损失: C_i * (effective_data)^(-β_i) + E_i
        domain_loss = C_i * (effective_data ** (-beta_i)) + E_i
        
        # 累加到总损失
        total_loss += domain_loss
    
    return total_loss

# ============================================================================
# 约束条件定义
# ============================================================================

# 权重边界约束: 0 ≤ w_i ≤ 1 for all i
bounds = Bounds(
    lb=[0.0, 0.0, 0.0],  # 下界
    ub=[1.0, 1.0, 1.0]   # 上界
)

# 等式约束: Σw_i = 1
equality_constraint = {
    'type': 'eq',
    'fun': lambda w: np.sum(w) - 1
}

print("约束条件:")
print(f"  权重边界: 0 ≤ w_i ≤ 1 for all i")
print(f"  等式约束: Σw_i = 1")
print()

# ============================================================================
# 优化设置
# ============================================================================

# 初始猜测: 均匀分布的权重
x0 = np.array([1/3, 1/3, 1/3])

print("优化设置:")
print(f"  初始权重: {x0}")
print(f"  优化算法: SLSQP (Sequential Least Squares Programming)")
print(f"  初始目标函数值: {objective_function(x0):.6f}")
print()

# ============================================================================
# 执行优化
# ============================================================================

print("开始优化...")
print("=" * 50)

# 使用SLSQP算法进行优化
result = minimize(
    objective_function,
    x0,
    method='SLSQP',
    bounds=bounds,
    constraints=equality_constraint,
    options={
        'disp': True,
        'maxiter': 1000,
        'ftol': 1e-9
    }
)

print("=" * 50)
print("优化完成!")
print()

# ============================================================================
# 结果输出
# ============================================================================

print("优化结果:")
print(f"  优化成功: {'是' if result.success else '否'}")
print(f"  收敛状态: {result.message}")
print(f"  迭代次数: {result.nit}")
print(f"  函数评估次数: {result.nfev}")
print()

print(f"最终最小化总损失: {result.fun:.8f}")
print()

# 最优权重
optimal_weights = result.x
print("最优权重分配:")
for i, domain in enumerate(domains):
    weight_decimal = optimal_weights[i]
    weight_percentage = weight_decimal * 100
    data_allocation = weight_decimal * N_0
    print(f"  {domain.capitalize():<8}: {weight_decimal:.6f} ({weight_percentage:.2f}%) = {data_allocation:,.0f} tokens")

print()
print(f"权重总和验证: {np.sum(optimal_weights):.8f} (应该等于 1.000000)")
weight_sum_check = abs(np.sum(optimal_weights) - 1.0) < 1e-6
print(f"权重总和检查: {'通过' if weight_sum_check else '失败'}")
print()

# ============================================================================
# 详细分析
# ============================================================================

print("详细损失分析:")
print(f"{'领域':<8} {'权重':<10} {'数据量(M)':<12} {'预测损失':<12} {'改进':<10}")
print("-" * 60)

# 计算初始损失（均匀分布）
initial_loss_per_domain = []
for i, domain in enumerate(domains):
    w_uniform = np.array([1/3, 1/3, 1/3])
    # 计算单个领域的损失
    params = parameters[domain]
    C_i, k_i, alpha_i, beta_i, E_i = params["C"], params["k"], params["alpha"], params["beta"], params["E"]
    w_i = w_uniform[i]
    domain_data = w_i * N_0
    other_data = N_0 - w_i * N_0
    effective_data = domain_data + k_i * (other_data ** alpha_i)
    domain_loss = C_i * (effective_data ** (-beta_i)) + E_i
    initial_loss_per_domain.append(domain_loss)

# 计算优化后的损失
optimal_loss_per_domain = []
for i, domain in enumerate(domains):
    params = parameters[domain]
    C_i, k_i, alpha_i, beta_i, E_i = params["C"], params["k"], params["alpha"], params["beta"], params["E"]
    w_i = optimal_weights[i]
    domain_data = w_i * N_0
    other_data = N_0 - w_i * N_0
    effective_data = domain_data + k_i * (other_data ** alpha_i)
    domain_loss = C_i * (effective_data ** (-beta_i)) + E_i
    optimal_loss_per_domain.append(domain_loss)

for i, domain in enumerate(domains):
    weight = optimal_weights[i]
    data_mb = (weight * N_0) / 1_000_000
    loss = optimal_loss_per_domain[i]
    improvement = initial_loss_per_domain[i] - loss
    improvement_pct = (improvement / initial_loss_per_domain[i]) * 100
    print(f"{domain.capitalize():<8} {weight:<10.4f} {data_mb:<12.1f} {loss:<12.6f} {improvement_pct:+.2f}%")

print("-" * 60)
total_initial_loss = sum(initial_loss_per_domain)
total_optimal_loss = sum(optimal_loss_per_domain)
total_improvement = total_initial_loss - total_optimal_loss
total_improvement_pct = (total_improvement / total_initial_loss) * 100

print(f"{'总计':<8} {1.0:<10.4f} {120.0:<12.1f} {total_optimal_loss:<12.6f} {total_improvement_pct:+.2f}%")
print()

print("优化总结:")
print(f"  初始总损失 (均匀分布): {total_initial_loss:.8f}")
print(f"  优化后总损失: {total_optimal_loss:.8f}")
print(f"  总改进: {total_improvement:.8f} ({total_improvement_pct:.2f}%)")
print()

if result.success:
    print("✅ 数据配比优化成功完成!")
    print(f"   推荐的最优数据配比为: Math {optimal_weights[0]*100:.1f}%, Code {optimal_weights[1]*100:.1f}%, Science {optimal_weights[2]*100:.1f}%")
else:
    print("❌ 优化未能收敛，请检查参数设置。")