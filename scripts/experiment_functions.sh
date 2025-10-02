#!/bin/bash

# SFT实验通用函数库
# 该文件只包含函数定义，不包含执行逻辑，可以安全地被其他脚本source

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

# 分布式评估配置
NUM_GPUS=${NUM_GPUS:-7}  # 默认使用8个GPU，可通过环境变量覆盖

# 时间统计变量
SCRIPT_START_TIME=$(date +%s)
TOTAL_TRAINING_TIME=0
TOTAL_EVALUATION_TIME=0

# 时间格式化函数
format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    
    if [ $hours -gt 0 ]; then
        printf "%d小时%d分钟%d秒" $hours $minutes $secs
    elif [ $minutes -gt 0 ]; then
        printf "%d分钟%d秒" $minutes $secs
    else
        printf "%d秒" $secs
    fi
}

# 检查Python环境
check_python_env() {
    print_info "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未找到，请确保已安装Python3"
        return 1
    fi
    
    # 检查必要的包
    python3 -c "import torch, transformers, trl, datasets, accelerate" 2>/dev/null || {
        print_error "缺少必要的Python包，请运行: pip install -r requirements.txt"
        return 1
    }
    
    print_success "Python环境检查通过"
    return 0
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

# 步骤2: 训练和评估（优化存储版本）
train_and_evaluate_models() {
    print_info "=== 步骤2: 训练和评估（优化存储版本） ==="
    
    # 清空之前的结果文件
    if [ -f "results.csv" ]; then
        rm results.csv
        print_info "清空之前的结果文件"
    fi
    
    # 获取所有训练数据集和验证数据集
    training_datasets=($(ls data/training/))
    validation_datasets=($(ls data/validation/))
    
    print_info "发现 ${#training_datasets[@]} 个训练数据集和 ${#validation_datasets[@]} 个验证数据集"
    
    total_evaluations=$((${#training_datasets[@]} * ${#validation_datasets[@]}))
    current_evaluation=0
    
    for dataset in "${training_datasets[@]}"; do
        print_info "开始训练模型: $dataset"
        
        output_dir="/home/tiger/.cache/outputs/model_on_${dataset}"
        
        # 删除可能存在的旧模型
        if [ -d "$output_dir" ]; then
            print_info "删除旧模型: $output_dir"
            rm -rf "$output_dir"
        fi
        
        # 开始分布式训练
        print_info "使用 $NUM_GPUS 个GPU进行分布式训练"
        
        # 设置分布式训练参数
        export MASTER_ADDR=localhost
        export MASTER_PORT=17238
        export WORLD_SIZE=$NUM_GPUS
        export RANK=0
        
        # 记录训练开始时间
        training_start_time=$(date +%s)
        
        # 使用torchrun启动分布式训练
        torchrun --nproc_per_node=$NUM_GPUS \
                 --master_port=17238 \
                 src/train.py \
            --dataset_path "data/training/$dataset" \
            --output_dir "$output_dir" \
            --run_name "sft-$dataset"
        
        # 计算训练时间
        training_end_time=$(date +%s)
        training_duration=$((training_end_time - training_start_time))
        TOTAL_TRAINING_TIME=$((TOTAL_TRAINING_TIME + training_duration))
        
        if [ $? -ne 0 ]; then
            print_error "模型 $dataset 训练失败"
            return 1
        fi
        
        print_success "模型 $dataset 训练完成 (用时: $(format_time $training_duration))"
        
        # 使用批量评估功能，一次性在所有验证集上评估这个模型
        print_info "开始批量评估模型 $dataset 在所有验证集上的性能"
        
        # 构建验证集路径数组
        validation_paths=()
        validation_names=()
        for val_dataset in "${validation_datasets[@]}"; do
            validation_paths+=("data/validation/$val_dataset")
            validation_names+=("$val_dataset")
        done
        
        # 记录评估开始时间
        evaluation_start_time=$(date +%s)
        
        # 使用torchrun启动分布式评估
        print_info "使用 $NUM_GPUS 个GPU进行分布式评估"
        torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --master_port=13238 --node_rank=0 \
            src/evaluate.py \
            --config "configs/training_config_thoth.yaml" \
            --model_path "$output_dir" \
            --train_dataset_name "$dataset" \
            --dataset_paths "${validation_paths[@]}" \
            --validation_dataset_names "${validation_names[@]}" \
            --output_file "results.csv"
        
        # 计算评估时间
        evaluation_end_time=$(date +%s)
        evaluation_duration=$((evaluation_end_time - evaluation_start_time))
        TOTAL_EVALUATION_TIME=$((TOTAL_EVALUATION_TIME + evaluation_duration))
        
        if [ $? -eq 0 ]; then
            print_success "批量评估完成: $dataset -> 所有验证集 (用时: $(format_time $evaluation_duration))"
        else
            print_error "批量评估失败: $dataset"
            return 1
        fi
        
        # 评估完成后删除模型checkpoint以节省存储空间
        print_info "删除模型checkpoint以节省存储空间: $output_dir"
        rm -rf "$output_dir"
        print_success "模型 $dataset 的所有评估完成，checkpoint已删除"
    done
    
    print_success "所有模型训练和评估完成"
}

# 显示结果摘要
show_results() {
    print_info "=== 实验结果摘要 ==="
    
    # 计算总用时
    script_end_time=$(date +%s)
    total_script_time=$((script_end_time - SCRIPT_START_TIME))
    
    if [ -f "results.csv" ]; then
        print_info "结果已保存到 results.csv"
        print_info "前10行结果预览:"
        head -n 11 results.csv | column -t -s ','
        
        total_results=$(tail -n +2 results.csv | wc -l)
        print_success "总共完成 $total_results 个评估实验"
    else
        print_error "未找到结果文件 results.csv"
    fi
    
    # 显示时间统计
    print_info "=== 时间统计 ==="
    print_info "总训练时间: $(format_time $TOTAL_TRAINING_TIME)"
    print_info "总评估时间: $(format_time $TOTAL_EVALUATION_TIME)"
    print_info "脚本总用时: $(format_time $total_script_time)"
    
    # 计算平均时间（如果有结果的话）
    if [ -f "results.csv" ] && [ $total_results -gt 0 ]; then
        training_datasets_count=$(ls data/training/ | wc -l)
        if [ $training_datasets_count -gt 0 ]; then
            avg_training_time=$((TOTAL_TRAINING_TIME / training_datasets_count))
            avg_evaluation_time=$((TOTAL_EVALUATION_TIME / training_datasets_count))
            print_info "平均每个模型训练时间: $(format_time $avg_training_time)"
            print_info "平均每个模型评估时间: $(format_time $avg_evaluation_time)"
        fi
    fi
}

# 清理函数
cleanup() {
    print_info "清理临时文件..."
    # 这里可以添加清理逻辑
}