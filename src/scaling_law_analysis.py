#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scaling Law分析脚本

该脚本基于loss_prediction_optimization.py，处理results.csv数据，
为每个验证数据集计算scaling law参数。

作者: AI Assistant
日期: 2024
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# 常量定义
# ============================================================================

# 基础数据量 (根据原始脚本)
N_BASE = 660000       # 基础token数
N_OTHER = 1320000     # 其他领域数据的总token数

# Huber损失参数
DELTA = 0.001

# 数据集变种映射
VARIANT_MULTIPLIERS = {
    '1_3': 1/3,
    '1_2': 1/2,
    '2x': 2,
    '3x': 3,
    'full_dataset': 1  # 对应1x
}

# ============================================================================
# 核心函数定义
# ============================================================================

def predict_loss(params, N_data_t):
    """
    根据损失预测模型计算预测损失
    
    公式: L_tilde(N_data^(t)) = C * (N_data^(t) + k * |N - N_data^(0)|^alpha)^(-beta) + E
    
    参数:
        params: [C, k, alpha, beta, E] 参数数组
        N_data_t: 第t次实验中数据集的token数量
    
    返回:
        预测的损失值
    """
    C, k, alpha, beta, E = params
    
    # 计算有效数据量
    effective_data = N_data_t + k * (N_OTHER ** alpha)
    
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
        delta: Huber损失的阈值参数
    
    返回:
        Huber损失值
    """
    abs_residual = abs(residual)
    if abs_residual <= delta:
        return 0.5 * residual**2
    else:
        return delta * (abs_residual - 0.5 * delta)

def objective_func(params, experimental_data):
    """
    目标函数：计算总Huber损失
    
    参数:
        params: [C, k, alpha, beta, E] 参数数组
        experimental_data: 实验数据列表 [(N_data_t, L_observed), ...]
    
    返回:
        总Huber损失
    """
    total_loss = 0.0
    
    for N_data_t, L_observed in experimental_data:
        L_predicted = predict_loss(params, N_data_t)
        residual = L_predicted - L_observed
        total_loss += huber_loss(residual, DELTA)
    
    return total_loss

def constraint_func(params):
    """
    约束函数：确保 k * |N - N_data|^alpha <= |N - N_data|
    
    参数:
        params: [C, k, alpha, beta, E] 参数数组
    
    返回:
        约束函数值 (应该 <= 0)
    """
    C, k, alpha, beta, E = params
    
    # 约束: k * |N - N_data|^alpha <= |N - N_data|
    # 重写为: k * |N - N_data|^alpha - |N - N_data| <= 0
    constraint_value = k * (N_OTHER ** alpha) - N_OTHER
    
    return constraint_value

def extract_base_dataset_name(train_dataset_name):
    """
    从训练数据集名称中提取基础数据集名称
    
    参数:
        train_dataset_name: 训练数据集名称
    
    返回:
        基础数据集名称和变种类型
    """
    if train_dataset_name == 'full_dataset':
        return 'full_dataset', 'full_dataset'
    
    # 检查各种变种后缀
    for variant in ['_1_3', '_1_2', '_2x', '_3x']:
        if train_dataset_name.endswith(variant):
            base_name = train_dataset_name[:-len(variant)]
            return base_name, variant[1:]  # 去掉前面的下划线
    
    # 如果没有匹配的变种，返回原名称
    return train_dataset_name, 'unknown'

def optimize_scaling_law(experimental_data, dataset_name):
    """
    为给定的实验数据优化scaling law参数
    
    参数:
        experimental_data: 实验数据列表 [(N_data_t, L_observed), ...]
        dataset_name: 数据集名称
    
    返回:
        优化结果字典
    """
    print(f"\n正在优化数据集: {dataset_name}")
    print(f"数据点数量: {len(experimental_data)}")
    
    # 初始参数 (基于原始脚本)
    x0 = np.array([0.9820, 0.1235, 0.5235, 0.0439, 1.2679])
    
    # 参数边界
    bounds = [
        (1e-6, 1.5),    # C > 0
        (1e-6, 0.3),    # k > 0  
        (1e-6, 0.8),    # alpha ∈ [-1, 0.8]
        (0.01, 0.1),    # beta > 0
        (0.0, 2.0)     # E 无限制
    ]
    
    # 非线性约束
    nonlinear_constraint = NonlinearConstraint(
        constraint_func, 
        -np.inf, 
        0.0
    )
    
    # 优化
    try:
        result = minimize(
            lambda params: objective_func(params, experimental_data),
            x0,
            method='trust-constr',
            bounds=bounds,
            constraints=nonlinear_constraint,
            options={
                'verbose': 0,
                'maxiter': 20000,
                'gtol': 1e-6,
                'xtol': 1e-6
            }
        )
        
        if result.success:
            C_opt, k_opt, alpha_opt, beta_opt, E_opt = result.x
            
            # 计算拟合质量指标
            total_huber = 0.0
            predictions = []
            residuals = []
            
            for N_data_t, L_observed in experimental_data:
                L_predicted = predict_loss(result.x, N_data_t)
                residual = L_predicted - L_observed
                huber = huber_loss(residual, DELTA)
                total_huber += huber
                predictions.append(L_predicted)
                residuals.append(residual)
            
            return {
                'dataset_name': dataset_name,
                'success': True,
                'C': C_opt,
                'k': k_opt,
                'alpha': alpha_opt,
                'beta': beta_opt,
                'E': E_opt,
                'total_huber_loss': total_huber,
                'mean_absolute_error': np.mean(np.abs(residuals)),
                'rmse': np.sqrt(np.mean(np.array(residuals)**2)),
                'data_points': len(experimental_data),
                'message': result.message
            }
        else:
            print(f"  优化失败: {result.message}")
            return {
                'dataset_name': dataset_name,
                'success': False,
                'C': np.nan,
                'k': np.nan,
                'alpha': np.nan,
                'beta': np.nan,
                'E': np.nan,
                'total_huber_loss': np.nan,
                'mean_absolute_error': np.nan,
                'rmse': np.nan,
                'data_points': len(experimental_data),
                'message': result.message
            }
    
    except Exception as e:
        print(f"  优化出错: {str(e)}")
        return {
            'dataset_name': dataset_name,
            'success': False,
            'C': np.nan,
            'k': np.nan,
            'alpha': np.nan,
            'beta': np.nan,
            'E': np.nan,
            'total_huber_loss': np.nan,
            'mean_absolute_error': np.nan,
            'rmse': np.nan,
            'data_points': len(experimental_data),
            'message': str(e)
        }

def main():
    """
    主函数：处理CSV数据并计算scaling law参数
    """
    print("Scaling Law分析脚本")
    print("="*60)
    
    # 读取CSV文件
    csv_path = '/Users/bytedance/Desktop/Github/sft_trainer/results.csv'
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"读取CSV文件: {csv_path}")
    print(f"数据行数: {len(df)}")
    
    # 获取所有唯一的验证数据集
    validation_datasets = sorted(df['validation_dataset_name'].unique())
    print(f"验证数据集数量: {len(validation_datasets)}")
    
    # 为每个验证数据集收集数据
    results = []
    
    for val_dataset in validation_datasets:
        print(f"\n处理验证数据集: {val_dataset}")
        
        # 获取该验证数据集的所有数据
        val_data = df[df['validation_dataset_name'] == val_dataset]
        
        # 收集实验数据
        experimental_data = []
        variant_data = {}
        
        for _, row in val_data.iterrows():
            train_dataset = row['train_dataset_name']
            loss = row['loss']
            
            base_name, variant = extract_base_dataset_name(train_dataset)
            
            if variant in VARIANT_MULTIPLIERS:
                multiplier = VARIANT_MULTIPLIERS[variant]
                N_data_t = N_BASE * multiplier
                experimental_data.append((N_data_t, loss))
                variant_data[variant] = (N_data_t, loss)
        
        print(f"  找到 {len(experimental_data)} 个数据点")
        print(f"  变种: {list(variant_data.keys())}")
        
        # 只有当我们有足够的数据点时才进行优化
        if len(experimental_data) >= 3:
            result = optimize_scaling_law(experimental_data, val_dataset)
            results.append(result)
            
            if result['success']:
                print(f"  ✅ 优化成功")
                print(f"     C={result['C']:.6f}, k={result['k']:.6f}, α={result['alpha']:.6f}")
                print(f"     β={result['beta']:.6f}, E={result['E']:.6f}")
                print(f"     Huber损失={result['total_huber_loss']:.8f}")
            else:
                print(f"  ❌ 优化失败: {result['message']}")
        else:
            print(f"  ⚠️  数据点不足 ({len(experimental_data)} < 3)，跳过")
    
    # 保存结果到CSV
    if results:
        results_df = pd.DataFrame(results)
        output_path = '../scaling_law_parameters.csv'
        results_df.to_csv(output_path, index=False)
        
        print(f"\n结果已保存到: {output_path}")
        print(f"成功优化的数据集数量: {sum(1 for r in results if r['success'])}")
        print(f"总数据集数量: {len(results)}")
        
        # 显示成功的结果摘要
        successful_results = [r for r in results if r['success']]
        if successful_results:
            print("\n成功优化的数据集摘要:")
            print(f"{'数据集':<40} {'C':<10} {'k':<10} {'α':<10} {'β':<10} {'E':<10} {'Huber损失':<12}")
            print("-" * 110)
            for r in successful_results:
                print(f"{r['dataset_name']:<40} {r['C']:<10.4f} {r['k']:<10.4f} {r['alpha']:<10.4f} {r['beta']:<10.4f} {r['E']:<10.4f} {r['total_huber_loss']:<12.6f}")
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()