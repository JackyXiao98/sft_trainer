#!/bin/bash

# SFT Scaling Law 实验自动化脚本
# 该脚本将执行完整的实验流程：数据生成、模型训练和评估

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python环境
check_python_env() {
    print_info "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未找到，请确保已安装Python3"
        exit 1
    fi
    
    # 检查必要的包
    python3 -c "import torch, transformers, trl, datasets, accelerate" 2>/dev/null || {
        print_error "缺少必要的Python包，请运行: pip install -r requirements.txt"
        exit 1
    }
    
    print_success "Python环境检查通过"
}

# 检查CUDA环境
check_cuda() {
    print_info "检查CUDA环境..."
    
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        print_success "CUDA环境可用"
    else
        print_warning "未检测到CUDA环境，将使用CPU训练（速度较慢）"
    fi
}

# 检查wandb登录状态
check_wandb() {
    print_info "检查Wandb登录状态..."
    
    if python3 -c "import wandb; wandb.api.api_key" 2>/dev/null; then
        print_success "Wandb已登录"
    else
        print_warning "Wandb未登录，请运行: wandb login"
        print_warning "或者在configs/training_config.yaml中将report_to设置为空列表"
    fi
}

# 步骤1: 数据生成
generate_datasets() {
    print_info "=== 步骤1: 生成数据集 ==="
    
    if [ -d "data/training" ] && [ -d "data/validation" ] && [ "$(ls -A data/training 2>/dev/null)" ] && [ "$(ls -A data/validation 2>/dev/null)" ]; then
        print_warning "数据集已存在，跳过数据生成步骤"
        print_info "如需重新生成数据集，请删除data目录后重新运行"
        return 0
    fi
    
    print_info "开始生成训练和验证数据集..."
    python3 src/data_builder.py
    
    if [ $? -eq 0 ]; then
        print_success "数据集生成完成"
    else
        print_error "数据集生成失败"
        exit 1
    fi
}

# 步骤2: 模型训练
train_models() {
    print_info "=== 步骤2: 模型训练 ==="
    
    # 获取所有训练数据集
    training_datasets=($(ls data/training/))
    
    print_info "发现 ${#training_datasets[@]} 个训练数据集"
    
    for dataset in "${training_datasets[@]}"; do
        print_info "开始训练模型: $dataset"
        
        output_dir="outputs/model_on_${dataset}"
        
        # 检查模型是否已经训练过
        if [ -d "$output_dir" ] && [ -f "$output_dir/pytorch_model.bin" -o -f "$output_dir/model.safetensors" ]; then
            print_warning "模型 $dataset 已存在，跳过训练"
            continue
        fi
        
        # 开始训练
        python3 src/train.py \
            --dataset_path "data/training/$dataset" \
            --output_dir "$output_dir" \
            --run_name "sft-$dataset"
        
        if [ $? -eq 0 ]; then
            print_success "模型 $dataset 训练完成"
        else
            print_error "模型 $dataset 训练失败"
            exit 1
        fi
    done
    
    print_success "所有模型训练完成"
}

# 步骤3: 模型评估
evaluate_models() {
    print_info "=== 步骤3: 模型评估 ==="
    
    # 清空之前的结果文件
    if [ -f "results.csv" ]; then
        rm results.csv
        print_info "清空之前的结果文件"
    fi
    
    # 获取所有训练数据集和验证数据集
    training_datasets=($(ls data/training/))
    validation_datasets=($(ls data/validation/))
    
    print_info "开始评估 ${#training_datasets[@]} 个训练模型在 ${#validation_datasets[@]} 个验证集上的性能"
    
    total_evaluations=$((${#training_datasets[@]} * ${#validation_datasets[@]}))
    current_evaluation=0
    
    for train_dataset in "${training_datasets[@]}"; do
        model_path="outputs/model_on_${train_dataset}"
        
        # 检查模型是否存在
        if [ ! -d "$model_path" ]; then
            print_error "模型 $model_path 不存在，跳过评估"
            continue
        fi
        
        for val_dataset in "${validation_datasets[@]}"; do
            current_evaluation=$((current_evaluation + 1))
            
            print_info "[$current_evaluation/$total_evaluations] 评估模型 $train_dataset 在验证集 $val_dataset 上的性能"
            
            python3 src/evaluate.py \
                --model_path "$model_path" \
                --dataset_path "data/validation/$val_dataset" \
                --train_dataset_name "$train_dataset" \
                --validation_dataset_name "$val_dataset" \
                --output_file "results.csv"
            
            if [ $? -eq 0 ]; then
                print_success "评估完成: $train_dataset -> $val_dataset"
            else
                print_error "评估失败: $train_dataset -> $val_dataset"
                exit 1
            fi
        done
    done
    
    print_success "所有模型评估完成"
}

# 显示结果摘要
show_results() {
    print_info "=== 实验结果摘要 ==="
    
    if [ -f "results.csv" ]; then
        print_info "结果已保存到 results.csv"
        print_info "前10行结果预览:"
        head -n 11 results.csv | column -t -s ','
        
        total_results=$(tail -n +2 results.csv | wc -l)
        print_success "总共完成 $total_results 个评估实验"
    else
        print_error "未找到结果文件 results.csv"
    fi
}

# 清理函数
cleanup() {
    print_info "清理临时文件..."
    # 这里可以添加清理逻辑
}

# 主函数
main() {
    print_info "开始SFT Scaling Law实验"
    print_info "实验将包括: 数据生成 -> 模型训练 -> 模型评估"
    
    # 环境检查
    check_python_env
    check_cuda
    check_wandb
    
    # 创建必要的目录
    mkdir -p outputs
    mkdir -p data/training
    mkdir -p data/validation
    
    # 执行实验步骤
    generate_datasets
    train_models
    evaluate_models
    show_results
    
    print_success "SFT Scaling Law实验完成!"
    print_info "请查看 results.csv 文件获取详细结果"
    print_info "训练好的模型保存在 outputs/ 目录下"
}

# 信号处理
trap cleanup EXIT

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-only)
            print_info "仅生成数据集"
            check_python_env
            generate_datasets
            exit 0
            ;;
        --train-only)
            print_info "仅执行训练"
            check_python_env
            check_cuda
            check_wandb
            train_models
            exit 0
            ;;
        --eval-only)
            print_info "仅执行评估"
            check_python_env
            evaluate_models
            show_results
            exit 0
            ;;
        --help|-h)
            echo "SFT Scaling Law 实验脚本"
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --data-only    仅生成数据集"
            echo "  --train-only   仅执行训练"
            echo "  --eval-only    仅执行评估"
            echo "  --help, -h     显示此帮助信息"
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            print_info "使用 --help 查看可用选项"
            exit 1
            ;;
    esac
    shift
done

# 执行主函数
main