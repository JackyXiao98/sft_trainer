#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13个数据集的数据配比优化脚本

基于scaling law参数，为13个数据集计算最优的数据配比。
使用SLSQP算法在总数据预算约束下最小化总预测损失。

新功能：数据集权重支持
- 支持为每个数据集设置重要性权重
- 在计算总损失时，每个数据集的损失会乘以对应的权重
- 默认所有数据集权重为1（同等重要性）
- 可以通过修改main()函数中的dataset_weights列表来调整权重

输入: scaling_law_parameters.csv (包含13个数据集的scaling law参数)
输出: dataset_mixing_optimization_results.csv (详细配比结果，包含数据集权重)
     optimization_summary.csv (优化摘要)

使用示例：
- 权重为1.0：正常重要性
- 权重为2.0：双倍重要性（该数据集的损失在优化中权重更大）
- 权重为0.5：较低重要性（该数据集的损失在优化中权重较小）

作者: AI Assistant
日期: 2024
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# 常量和参数定义
# ============================================================================

# 总数据预算 (120 Million tokens)
N_0 = 120_000_000

# ============================================================================
# 核心优化函数
# ============================================================================

def objective_function(w, datasets_params, dataset_weights=None):
    """
    目标函数: 计算所有数据集预测损失的加权总和
    
    公式: min_w Σ[weight_i * (C_i * (w_i*N_0 + k_i*(N_0 - w_i*N_0)^α_i)^(-β_i) + E_i)]
    
    参数:
        w: 权重向量 [w_1, w_2, ..., w_13] (13个数据集的权重)
        datasets_params: 包含13个数据集scaling law参数的列表
        dataset_weights: 数据集重要性权重列表，默认为None（所有权重为1）
    
    返回:
        总预测损失
    """
    total_loss = 0.0
    n_datasets = len(datasets_params)
    
    # 如果没有提供数据集权重，默认所有权重为1
    if dataset_weights is None:
        dataset_weights = [1.0] * n_datasets
    
    for i in range(n_datasets):
        # 获取当前数据集的参数
        params = datasets_params[i]
        C_i = params["C"]
        k_i = params["k"]
        alpha_i = params["alpha"]
        beta_i = params["beta"]
        E_i = params["E"]
        
        # 获取当前数据集的权重
        w_i = w[i]
        
        # 获取当前数据集的重要性权重
        weight_i = dataset_weights[i]
        
        # 计算当前数据集的数据量
        dataset_data = w_i * N_0
        other_data = N_0 - w_i * N_0
        
        # 计算有效数据量: w_i*N_0 + k_i*(N_0 - w_i*N_0)^α_i
        effective_data = dataset_data + k_i * (other_data ** alpha_i)
        
        # 避免数值问题
        if effective_data <= 0:
            return float('inf')
        
        # 计算当前数据集的预测损失: C_i * (effective_data)^(-β_i) + E_i
        dataset_loss = C_i * (effective_data ** (-beta_i)) + E_i
        
        # 将损失乘以数据集权重后累加到总损失
        total_loss += weight_i * dataset_loss
    
    return total_loss

def optimize_13_datasets_mixing(datasets_params, dataset_names, dataset_weights=None):
    """
    为13个数据集优化数据配比
    
    参数:
        datasets_params: 包含13个数据集scaling law参数的列表
        dataset_names: 13个数据集名称的列表
        dataset_weights: 数据集重要性权重列表，默认为None（所有权重为1）
    
    返回:
        优化结果字典
    """
    n_datasets = len(datasets_params)
    
    # 权重边界约束: 0 ≤ w_i ≤ 1 for all i
    bounds = Bounds(
        lb=[0.0] * n_datasets,  # 下界
        ub=[1.0] * n_datasets   # 上界
    )
    
    # 等式约束: Σw_i = 1
    equality_constraint = {
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    }
    
    # 初始猜测: 均匀分布的权重
    x0 = np.array([1/n_datasets] * n_datasets)
    
    try:
        # 使用SLSQP算法进行优化
        result = minimize(
            lambda w: objective_function(w, datasets_params, dataset_weights),
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=equality_constraint,
            options={
                'disp': False,
                'maxiter': 10000,
                'ftol': 1e-9
            }
        )
        
        if result.success:
            optimal_weights = result.x
            
            # 计算初始损失（均匀分布）
            initial_loss = objective_function(x0, datasets_params, dataset_weights)
            optimal_loss = result.fun
            improvement = initial_loss - optimal_loss
            improvement_pct = (improvement / initial_loss) * 100 if initial_loss > 0 else 0
            
            # 构建结果字典
            result_dict = {
                'success': True,
                'total_loss': optimal_loss,
                'initial_loss': initial_loss,
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'iterations': result.nit,
                'function_evals': result.nfev,
                'message': result.message
            }
            
            # 添加每个数据集的权重和token数量
            for i, dataset_name in enumerate(dataset_names):
                result_dict[f'{dataset_name}_weight'] = optimal_weights[i]
                result_dict[f'{dataset_name}_tokens'] = optimal_weights[i] * N_0
            
            return result_dict
            
        else:
            result_dict = {
                'success': False,
                'total_loss': np.nan,
                'initial_loss': np.nan,
                'improvement': np.nan,
                'improvement_pct': np.nan,
                'iterations': result.nit if hasattr(result, 'nit') else 0,
                'function_evals': result.nfev if hasattr(result, 'nfev') else 0,
                'message': result.message
            }
            
            # 添加每个数据集的NaN值
            for dataset_name in dataset_names:
                result_dict[f'{dataset_name}_weight'] = np.nan
                result_dict[f'{dataset_name}_tokens'] = np.nan
            
            return result_dict
    
    except Exception as e:
        result_dict = {
            'success': False,
            'total_loss': np.nan,
            'initial_loss': np.nan,
            'improvement': np.nan,
            'improvement_pct': np.nan,
            'iterations': 0,
            'function_evals': 0,
            'message': str(e)
        }
        
        # 添加每个数据集的NaN值
        for dataset_name in dataset_names:
            result_dict[f'{dataset_name}_weight'] = np.nan
            result_dict[f'{dataset_name}_tokens'] = np.nan
        
        return result_dict

def main():
    """
    主函数：处理13个数据集的数据配比优化
    """
    print("13个数据集配比优化脚本")
    print("=" * 60)
    print(f"总数据预算: {N_0:,} tokens ({N_0/1_000_000:.0f}M tokens)")
    print()
    
    # 读取scaling law参数文件
    csv_path = '../scaling_law_parameters.csv'
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"读取scaling law参数文件: {csv_path}")
    print(f"数据集数量: {len(df)}")
    print()
    
    # 过滤成功的数据集
    successful_datasets = df[df['success'] == True]
    print(f"成功优化的数据集数量: {len(successful_datasets)}")
    print()
    
    # 准备13个数据集的参数和名称
    datasets_params = []
    dataset_names = []
    
    for idx, row in successful_datasets.iterrows():
        dataset_name = row['dataset_name']
        dataset_names.append(dataset_name)
        
        # 提取scaling law参数
        params = {
            "C": row['C'],
            "k": row['k'],
            "alpha": row['alpha'],
            "beta": row['beta'],
            "E": row['E']
        }
        datasets_params.append(params)
        
        print(f"数据集 {len(datasets_params)}: {dataset_name}")
        print(f"  参数: C={params['C']:.4f}, k={params['k']:.4f}, α={params['alpha']:.4f}, β={params['beta']:.4f}, E={params['E']:.4f}")
    
    print()
    
    # 配置数据集权重（可以根据需要调整）
    # 默认所有数据集权重为1，表示同等重要性
    # 如果需要调整某个数据集的重要性，可以修改对应的权重值
    # dataset_weights = [1.0] * len(dataset_names)  # 默认权重为1
    
    # 示例：如果想要某些数据集更重要，可以这样设置：
    dataset_weights = [
        1.0,  # 第1个数据集权重为2（更重要）
        1.0,  # 第2个数据集权重为1（正常）
        0.5,  # 第3个数据集权重为0.5（较不重要）
        0.5,
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
    ]
    
    print("数据集权重配置:")
    for i, (name, weight) in enumerate(zip(dataset_names, dataset_weights)):
        print(f"  {i+1:2d}. {name:<50} 权重: {weight:.1f}")
    
    print()
    print("开始优化13个数据集的配比...")
    
    # 优化13个数据集的配比
    result = optimize_13_datasets_mixing(datasets_params, dataset_names, dataset_weights)
    
    if result['success']:
        print("✅ 优化成功!")
        print(f"总损失: {result['total_loss']:.6f}")
        print(f"初始损失: {result['initial_loss']:.6f}")
        print(f"改进: {result['improvement_pct']:.2f}%")
        print(f"迭代次数: {result['iterations']}")
        print()
        
        print("各数据集的最优配比:")
        print(f"{'数据集名称':<50} {'权重':<10} {'Token数量':<15} {'百分比':<8}")
        print("-" * 90)
        
        total_weight = 0
        for dataset_name in dataset_names:
            weight = result[f'{dataset_name}_weight']
            tokens = result[f'{dataset_name}_tokens']
            percentage = weight * 100
            total_weight += weight
            print(f"{dataset_name:<50} {weight:<10.6f} {tokens:<15,.0f} {percentage:<8.2f}%")
        
        print("-" * 90)
        print(f"{'总计':<50} {total_weight:<10.6f} {N_0:<15,.0f} {100.0:<8.1f}%")
        
        # 保存结果到CSV
        # 创建一个更易读的结果DataFrame
        results_data = []
        for i, dataset_name in enumerate(dataset_names):
            results_data.append({
                'dataset_name': dataset_name,
                'weight': result[f'{dataset_name}_weight'],
                'tokens': result[f'{dataset_name}_tokens'],
                'percentage': result[f'{dataset_name}_weight'] * 100,
                'dataset_importance_weight': dataset_weights[i],  # 添加数据集重要性权重
                'C': datasets_params[i]['C'],
                'k': datasets_params[i]['k'],
                'alpha': datasets_params[i]['alpha'],
                'beta': datasets_params[i]['beta'],
                'E': datasets_params[i]['E']
            })
        
        # 添加优化摘要信息
        summary_data = {
            'total_loss': result['total_loss'],
            'initial_loss': result['initial_loss'],
            'improvement': result['improvement'],
            'improvement_pct': result['improvement_pct'],
            'iterations': result['iterations'],
            'function_evals': result['function_evals'],
            'success': result['success'],
            'message': result['message']
        }
        
        # 保存详细结果
        results_df = pd.DataFrame(results_data)
        output_path = '../dataset_mixing_optimization_results.csv'
        results_df.to_csv(output_path, index=False)
        
        # 保存优化摘要
        summary_df = pd.DataFrame([summary_data])
        summary_path = '../optimization_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n结果已保存到: {output_path}")
        print(f"优化摘要已保存到: {summary_path}")
        
    else:
        print("❌ 优化失败!")
        print(f"错误信息: {result['message']}")
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()