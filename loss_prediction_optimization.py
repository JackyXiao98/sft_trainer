#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
损失预测模型参数优化脚本

该脚本使用非线性约束优化来拟合损失预测模型的参数。
模型基于scaling law理论，通过最小化Huber损失来优化参数。

作者: AI Assistant
日期: 2024
"""

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 常量定义
# ============================================================================

# 基础数据量
N_CODE_BASE = 660000  # code数据集的基础token数 N_code^(0)
N_OTHER = 1320000     # 其他领域数据(math+science)的总token数 |N - N_code|

# Huber损失参数
DELTA = 0.001

# 实验数据
# 格式: (N_code_t, L_code_t_observed)
# experimental_data = [
#     # t=0: baseline (1x code)
#     (660000 * 1, 0.5655004955709927 + 0.7627777529485298 + 1.1549238205459047),
#     # t=1: code_1_3 (1/3x code)
#     (660000 * (1/3), 0.5609681899585421 + 0.8135941916643971 + 1.1464412223689164),
#     # t=2: code_1_2 (1/2x code)
#     (660000 * (1/2), 0.5338599962137994 + 0.7633443399511203 + 1.0913624213969155),
#     # t=3: code_2x (2x code)
#     (660000 * 2, 0.5811840244938457 + 0.7418261641504789 + 1.182430182328204),
#     # t=4: code_3x (3x code)
#     (660000 * 3, 0.5067948727380662 + 0.5935674406061269 + 1.0421503492422748)
# ]

# 实验数据
# 格式: (N_math_t, L_math_t_observed)
experimental_data = [
    # t=0: baseline (1x math) - SHARED DATA POINT
    (660000 * 1, 0.5655004955709927 + 0.7627777529485298 + 1.1549238205459047),
    # t=1: math_1_3 (1/3x math)
    (660000 * (1/3), 0.6099057570099831 + 0.774502798463359 + 1.1750610731070554),
    # t=2: math_1_2 (1/2x math)
    (660000 * (1/2), 0.5626428419398883 + 0.7156369591301138 + 1.0905987285863499),
    # t=3: math_2x (2x math)
    (660000 * 2, 0.5353056646528698 + 0.7620182660492983 + 1.1491871712580009),
    # t=4: math_3x (3x math)
    (660000 * 3, 0.43585690300142954 + 0.6760521982655381 + 1.0375796740693886)
]

print("实验数据:")
for i, (n_code, loss) in enumerate(experimental_data):
    print(f"  t={i}: N_code={n_code:,.0f}, L_observed={loss:.6f}")
print()

# ============================================================================
# 核心函数定义
# ============================================================================

def predict_loss(params, N_code_t):
    """
    根据损失预测模型计算预测损失
    
    公式: L_tilde(N_code^(t)) = C * (N_code^(t) + k * |N - N_code^(0)|^alpha)^(-beta) + E
    
    参数:
        params: [C, k, alpha, beta, E] 参数数组
        N_code_t: 第t次实验中code数据集的token数量
    
    返回:
        预测的损失值
    """
    C, k, alpha, beta, E = params
    
    # 计算有效数据量: N_code^(t) + k * |N - N_code^(0)|^alpha
    effective_data = N_code_t + k * (N_OTHER ** alpha)
    
    # 避免数值问题
    if effective_data <= 0:
        return float('inf')
    
    # 计算预测损失
    predicted_loss = C * (effective_data ** (-beta)) + E
    
    return predicted_loss

def huber_loss(residual, delta):
    """
    计算Huber损失
    
    参数:
        residual: 残差 (预测值 - 真实值)
        delta: Huber损失的参数
    
    返回:
        Huber损失值
    """
    abs_residual = abs(residual)
    
    if abs_residual <= delta:
        return 0.5 * residual**2
    else:
        return delta * (abs_residual - 0.5 * delta)

def objective_func(params):
    """
    目标函数: 计算所有数据点的Huber损失之和
    
    参数:
        params: [C, k, alpha, beta, E] 参数数组
    
    返回:
        总Huber损失
    """
    total_loss = 0.0
    
    for N_code_t, L_observed in experimental_data:
        # 计算预测损失
        L_predicted = predict_loss(params, N_code_t)
        
        # 计算残差
        residual = L_predicted - L_observed
        
        # 累加Huber损失
        total_loss += huber_loss(residual, DELTA)
    
    return total_loss

def constraint_func(params):
    """
    非线性约束函数
    
    约束: k * |N - N_code|^alpha <= |N - N_code|
    转换为: k * |N - N_code|^alpha - |N - N_code| <= 0
    
    参数:
        params: [C, k, alpha, beta, E] 参数数组
    
    返回:
        约束函数值 (应该 <= 0)
    """
    C, k, alpha, beta, E = params
    
    # 计算约束: k * |N - N_code|^alpha - |N - N_code|
    constraint_value = k * (N_OTHER ** alpha) - N_OTHER
    
    return constraint_value

# ============================================================================
# 优化设置
# ============================================================================

# 初始参数猜测 [C, k, alpha, beta, E]
x0 = np.array([0.9820, 0.1235, 0.5235, 0.0439, 1.2679])

# 参数边界 (C > 0, k > 0, alpha ∈ [-5, 5], beta > 0, E可以为任意值)
bounds = [
    (1e-6, 1.5),    # C > 0
    (1e-6, 0.3),    # k > 0  
    (-1.0, 0.8),     # alpha ∈ [-5, 5]
    (1e-6, 0.1),    # beta > 0
    (-2.0, 2.0)     # E 无限制
]

# 非线性约束
# constraint_func(params) <= 0
nonlinear_constraint = NonlinearConstraint(
    constraint_func, 
    -np.inf, 
    0.0
)

print("优化设置:")
print(f"  初始参数: {x0}")
print(f"  参数边界: {bounds}")
print(f"  约束: k * |N - N_code|^alpha <= |N - N_code|")
print(f"  优化方法: trust-constr")
print()

# ============================================================================
# 执行优化
# ============================================================================

print("开始优化...")
print("="*60)

# 使用trust-constr方法进行优化
result = minimize(
    objective_func,
    x0,
    method='trust-constr',
    bounds=bounds,
    constraints=nonlinear_constraint,
    options={
        'verbose': 1,
        'maxiter': 20000,
        'gtol': 1e-8,
        'xtol': 1e-8
    }
)

print("="*60)
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

print(f"最终目标函数值 (总Huber损失): {result.fun:.8f}")
print()

print("最佳参数值:")
optimal_params = result.x
C_opt, k_opt, alpha_opt, beta_opt, E_opt = optimal_params
print(f"  C = {C_opt:.6f}")
print(f"  k = {k_opt:.6f}")
print(f"  α = {alpha_opt:.6f}")
print(f"  β = {beta_opt:.6f}")
print(f"  E = {E_opt:.6f}")
print()

# 验证约束是否满足
constraint_value = constraint_func(optimal_params)
print(f"约束验证:")
print(f"  约束函数值: {constraint_value:.6f} (应该 <= 0)")
print(f"  约束满足: {'是' if constraint_value <= 1e-6 else '否'}")
print()

# ============================================================================
# 预测结果对比
# ============================================================================

print("预测结果对比:")
print(f"{'实验':<8} {'N_code':<12} {'观测损失':<12} {'预测损失':<12} {'残差':<12} {'Huber损失':<12}")
print("-" * 80)

total_huber = 0.0
for i, (N_code_t, L_observed) in enumerate(experimental_data):
    L_predicted = predict_loss(optimal_params, N_code_t)
    residual = L_predicted - L_observed
    huber = huber_loss(residual, DELTA)
    total_huber += huber
    
    print(f"t={i:<7} {N_code_t:<12.0f} {L_observed:<12.6f} {L_predicted:<12.6f} {residual:<12.6f} {huber:<12.8f}")

print("-" * 80)
print(f"{'总计':<8} {'':<12} {'':<12} {'':<12} {'':<12} {total_huber:<12.8f}")
print()

# ============================================================================
# 模型解释
# ============================================================================

print("模型解释:")
print(f"  损失预测公式: L̃ = {C_opt:.4f} * (N_code + {k_opt:.4f} * {N_OTHER}^{alpha_opt:.4f})^(-{beta_opt:.4f}) + {E_opt:.4f}")
print(f"  有效数据转移系数 k: {k_opt:.6f}")
print(f"  数据量缩放指数 α: {alpha_opt:.6f}")
print(f"  损失缩放指数 β: {beta_opt:.6f}")
print(f"  基础损失偏移 E: {E_opt:.6f}")
print()

if result.success:
    print("✅ 优化成功完成!")
else:
    print("❌ 优化未能收敛，请检查初始参数或约束设置。")