#!/bin/bash

# TikTok评论数据集 SFT Scaling Law 实验自动化脚本
# 该脚本将执行完整的实验流程：TikTok数据生成、模型训练和评估
# 
# 分布式训练和评估配置:
# - 使用 torchrun 启动分布式训练和评估，默认使用 8 个 GPU
# - 可通过环境变量 NUM_GPUS 调整 GPU 数量，例如: NUM_GPUS=4 ./run_tiktok_exp.sh


# 导入通用函数库（只包含函数定义，不会执行任何主逻辑）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_functions.sh"

# 手动设置GPU数量
NUM_GPUS=8

# 配置文件目录和当前配置变量
CONFIG_DIR="configs/tt_cmt"
CURRENT_CONFIG=""
CURRENT_CONFIG_NAME=""

# 重写数据生成函数以使用TikTok数据构建器
generate_tiktok_datasets() {
    print_info "=== 步骤1: 生成TikTok评论数据集 ==="
    
    if [ -d "/home/tiger/.cache/data/training" ] && [ -d "/home/tiger/.cache/data/validation" ] && [ "$(ls -A /home/tiger/.cache/data/training 2>/dev/null)" ] && [ "$(ls -A /home/tiger/.cache/data/validation 2>/dev/null)" ]; then
        print_warning "数据集已存在，跳过数据生成步骤"
        print_info "如需重新生成数据集，请删除/home/tiger/.cache/data目录后重新运行"
        return 0
    fi
    
    print_info "开始生成TikTok评论训练和验证数据集..."
    print_info "使用TikTokCommentDataBuilder处理41个区间数据集"
    print_info "将生成41个训练集和41个验证集"
    
    # 使用当前配置文件
    python3 src/data_split/tiktok_comment_data_builder.py --config "$CURRENT_CONFIG"
    
    if [ $? -eq 0 ]; then
        print_success "TikTok评论数据集生成完成"
        
        # 显示生成的数据集统计
        if [ -d "/home/tiger/.cache/data/training" ]; then
            training_count=$(ls /home/tiger/.cache/data/training/ | wc -l)
            print_info "生成的训练数据集数量: $training_count"
        fi
        
        if [ -d "/home/tiger/.cache/data/validation" ]; then
            validation_count=$(ls /home/tiger/.cache/data/validation/ | wc -l)
            print_info "生成的验证数据集数量: $validation_count"
        fi
        
        return 0
    else
        print_error "TikTok评论数据集生成失败"
        return 1
    fi
}

# 检查TikTok数据源是否可访问
check_tiktok_data_source() {
    print_info "检查TikTok评论数据源..."
    
    local base_dir="/mnt/hdfs/selection/tiktok_cmt"
    
    if [ ! -d "$base_dir" ]; then
        print_error "TikTok数据源目录不存在: $base_dir"
        print_error "请确保HDFS挂载正确或数据源路径正确"
        return 1
    fi
    
    # 检查一些关键区间目录是否存在
    local test_intervals=(
        "[0.0, 0.1)"
        "[0.1, 0.2)"
        "[0.2, 0.3)"
    )
    
    local found_intervals=0
    for interval in "${test_intervals[@]}"; do
        if [ -d "$base_dir/$interval" ]; then
            found_intervals=$((found_intervals + 1))
        fi
    done
    
    if [ $found_intervals -eq 0 ]; then
        print_error "未找到任何预期的TikTok区间目录"
        print_error "请检查数据源路径和目录结构是否正确"
        return 1
    fi
    
    print_success "TikTok数据源检查通过 (找到 $found_intervals/${#test_intervals[@]} 个测试区间)"
    return 0
}

# 检查TikTok处理依赖
check_tiktok_dependencies() {
    print_info "检查TikTok数据处理依赖..."
    
    # 检查必要的包
    python3 -c "import transformers, datasets, jinja2, concurrent.futures" 2>/dev/null || {
        print_error "缺少TikTok数据处理依赖，请运行: pip install transformers datasets jinja2"
        return 1
    }
    
    print_success "TikTok依赖检查通过"
    return 0
}

# 检查配置文件
check_tiktok_config() {
    print_info "检查TikTok实验配置文件: $CURRENT_CONFIG"
    
    if [ ! -f "$CURRENT_CONFIG" ]; then
        print_error "配置文件不存在: $CURRENT_CONFIG"
        return 1
    fi
    
    if ! grep -q "use_flash_attention_2.*true" "$CURRENT_CONFIG"; then
        print_warning "配置文件中未启用flash_attention_2"
    fi
    
    print_success "TikTok配置文件检查通过: $CURRENT_CONFIG"
    return 0
}

# 重写训练和评估函数以使用当前配置
train_and_evaluate_tiktok_models() {
    print_info "=== 步骤2: 训练和评估TikTok模型（使用配置: $CURRENT_CONFIG_NAME） ==="
    
    # 生成当前配置对应的结果文件名
    local results_file="results_${CURRENT_CONFIG_NAME}.csv"
    
    # 清空之前的结果文件
    if [ -f "$results_file" ]; then
        rm "$results_file"
        print_info "清空之前的结果文件: $results_file"
    fi
    
    # 获取所有训练数据集和验证数据集
    training_datasets=($(ls /home/tiger/.cache/data/training/))
    validation_datasets=($(ls /home/tiger/.cache/data/validation/))
    
    print_info "发现 ${#training_datasets[@]} 个训练数据集和 ${#validation_datasets[@]} 个验证数据集"
    
    total_evaluations=$((${#training_datasets[@]} * ${#validation_datasets[@]}))
    current_evaluation=0
    
    for dataset in "${training_datasets[@]}"; do
        print_info "开始训练模型: $dataset"
        
        output_dir="/home/tiger/.cache/outputs/tiktok_model_on_${dataset}"
        
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
        
        # 使用torchrun启动分布式训练，指定当前配置文件
        torchrun --nproc_per_node=$NUM_GPUS \
                 --master_port=17238 \
                 src/train.py \
            --dataset_path "/home/tiger/.cache/data/training/$dataset" \
            --output_dir "$output_dir" \
            --config "$CURRENT_CONFIG" \
            --run_name "tiktok-sft-${CURRENT_CONFIG_NAME}-$dataset"
        
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
            validation_paths+=("/home/tiger/.cache/data/validation/$val_dataset")
            validation_names+=("$val_dataset")
        done
        
        # 记录评估开始时间
        evaluation_start_time=$(date +%s)
        
        # 使用torchrun启动分布式评估
        print_info "使用 $NUM_GPUS 个GPU进行分布式评估"
        torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --master_port=13238 --node_rank=0 \
            src/evaluate.py \
            --config "$CURRENT_CONFIG" \
            --model_path "$output_dir" \
            --train_dataset_name "$dataset" \
            --dataset_paths "${validation_paths[@]}" \
            --validation_dataset_names "${validation_names[@]}" \
            --output_file "$results_file"
        
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
    
    print_success "所有TikTok模型训练和评估完成"
}

# 主函数 - 专门用于TikTok数据集实验
main_tiktok() {
    print_info "开始TikTok评论数据集 SFT Scaling Law实验"
    print_info "实验将包括: TikTok数据生成 -> 模型训练 -> 模型评估"
    print_info "预计生成41个训练集和41个验证集"
    
    # 获取所有配置文件
    config_files=($(ls "$CONFIG_DIR"/*.yaml 2>/dev/null))
    
    if [ ${#config_files[@]} -eq 0 ]; then
        print_error "在 $CONFIG_DIR 目录下未找到任何 YAML 配置文件"
        return 1
    fi
    
    print_info "找到 ${#config_files[@]} 个配置文件: ${config_files[*]}"
    
    # 环境检查
    check_cuda
    # check_wandb
    
    # 创建必要的目录
    mkdir -p outputs
    mkdir -p /home/tiger/.cache/data/training
    mkdir -p /home/tiger/.cache/data/validation
    
    # 对每个配置文件执行完整的实验流程
    for config_file in "${config_files[@]}"; do
        # 设置当前配置变量
        CURRENT_CONFIG="$config_file"
        CURRENT_CONFIG_NAME=$(basename "$config_file" .yaml)
        
        print_info "=========================================="
        print_info "开始处理配置文件: $CURRENT_CONFIG_NAME"
        print_info "配置文件路径: $CURRENT_CONFIG"
        print_info "=========================================="
        
        # 环境检查
        # if ! check_python_env; then
        #     print_error "Python环境检查失败，跳过配置: $CURRENT_CONFIG_NAME"
        #     continue
        # fi
        
        # if ! check_tiktok_dependencies; then
        #     print_error "TikTok依赖检查失败，跳过配置: $CURRENT_CONFIG_NAME"
        #     continue
        # fi
        
        if ! check_tiktok_config; then
            print_error "配置文件检查失败，跳过配置: $CURRENT_CONFIG_NAME"
            continue
        fi
        
        # if ! check_tiktok_data_source; then
        #     print_error "TikTok数据源检查失败，跳过配置: $CURRENT_CONFIG_NAME"
        #     continue
        # fi
        
        # 执行实验步骤
        if ! generate_tiktok_datasets; then
            print_error "TikTok数据生成失败，跳过配置: $CURRENT_CONFIG_NAME"
            continue
        fi
        
        if ! train_and_evaluate_tiktok_models; then
            print_error "训练和评估失败，跳过配置: $CURRENT_CONFIG_NAME"
            continue
        fi
        
        print_success "配置 $CURRENT_CONFIG_NAME 的实验完成!"
        print_info "结果已保存到: results_${CURRENT_CONFIG_NAME}.csv"
    done
    
    print_success "所有配置文件的TikTok评论数据集 SFT Scaling Law实验完成!"
    print_info "结果文件:"
    for config_file in "${config_files[@]}"; do
        config_name=$(basename "$config_file" .yaml)
        if [ -f "results_${config_name}.csv" ]; then
            print_info "  - results_${config_name}.csv"
        fi
    done
    
    # 最终时间统计
    final_end_time=$(date +%s)
    final_total_time=$((final_end_time - SCRIPT_START_TIME))
    print_info "实验总耗时: $(format_time $final_total_time)"
}

# 解析命令行参数 - 扩展原有功能
parse_tiktok_args() {
    while [[ $# -gt 0 ]]; do
    case $1 in
        --data-only)
            print_info "仅生成TikTok评论数据集（所有配置文件）"
            
            # 获取所有配置文件
            config_files=($(ls "$CONFIG_DIR"/*.yaml 2>/dev/null))
            
            if [ ${#config_files[@]} -eq 0 ]; then
                print_error "在 $CONFIG_DIR 目录下未找到任何 YAML 配置文件"
                return 1
            fi
            
            # 创建必要的目录
            mkdir -p /home/tiger/.cache/data/training
            mkdir -p /home/tiger/.cache/data/validation
            
            # 为每个配置文件生成数据
            for config_file in "${config_files[@]}"; do
                CURRENT_CONFIG="$config_file"
                CURRENT_CONFIG_NAME=$(basename "$config_file" .yaml)
                
                print_info "为配置 $CURRENT_CONFIG_NAME 生成数据..."
                
                if check_tiktok_config && generate_tiktok_datasets; then
                    print_success "配置 $CURRENT_CONFIG_NAME 的数据生成完成"
                else
                    print_error "配置 $CURRENT_CONFIG_NAME 的数据生成失败"
                fi
            done
            return 0
            ;;
        --train-only)
            print_info "仅执行训练（使用现有TikTok数据集，所有配置文件）"
            
            # 获取所有配置文件
            config_files=($(ls "$CONFIG_DIR"/*.yaml 2>/dev/null))
            
            if [ ${#config_files[@]} -eq 0 ]; then
                print_error "在 $CONFIG_DIR 目录下未找到任何 YAML 配置文件"
                return 1
            fi
            
            check_cuda
            # check_wandb
            
            # 为每个配置文件执行训练
            for config_file in "${config_files[@]}"; do
                CURRENT_CONFIG="$config_file"
                CURRENT_CONFIG_NAME=$(basename "$config_file" .yaml)
                
                print_info "使用配置 $CURRENT_CONFIG_NAME 执行训练..."
                
                if check_tiktok_config && train_and_evaluate_tiktok_models; then
                    print_success "配置 $CURRENT_CONFIG_NAME 的训练完成"
                else
                    print_error "配置 $CURRENT_CONFIG_NAME 的训练失败"
                fi
            done
            return 0
            ;;
        --eval-only)
            print_info "仅执行评估"
            if check_python_env; then
                show_results
            else
                print_error "Python环境检查失败"
            fi
            return 0
            ;;
        --check-data)
            print_info "仅检查TikTok数据源"
            if check_tiktok_data_source; then
                print_success "数据源检查通过"
            else
                print_error "数据源检查失败"
            fi
            return 0
            ;;
        --check-config)
            print_info "仅检查配置文件（所有配置文件）"
            
            # 获取所有配置文件
            config_files=($(ls "$CONFIG_DIR"/*.yaml 2>/dev/null))
            
            if [ ${#config_files[@]} -eq 0 ]; then
                print_error "在 $CONFIG_DIR 目录下未找到任何 YAML 配置文件"
                return 1
            fi
            
            all_passed=true
            
            # 检查每个配置文件
            for config_file in "${config_files[@]}"; do
                CURRENT_CONFIG="$config_file"
                CURRENT_CONFIG_NAME=$(basename "$config_file" .yaml)
                
                if check_tiktok_config; then
                    print_success "配置文件 $CURRENT_CONFIG_NAME 检查通过"
                else
                    print_error "配置文件 $CURRENT_CONFIG_NAME 检查失败"
                    all_passed=false
                fi
            done
            
            if [ "$all_passed" = true ]; then
                print_success "所有配置文件检查通过"
            else
                print_error "部分配置文件检查失败"
            fi
            return 0
            ;;
        --help|-h)
            echo "TikTok评论数据集 SFT Scaling Law 实验脚本（多配置文件版本）"
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --data-only      仅生成TikTok评论数据集"
            echo "  --train-only     仅执行训练"
            echo "  --eval-only      仅执行评估"
            echo "  --check-data     仅检查TikTok数据源"
            echo "  --check-config   仅检查配置文件"
            echo "  --help, -h       显示此帮助信息"
            echo ""
            echo "数据集信息:"
            echo "  - 源数据集: 41个TikTok评论区间数据集"
            echo "  - 训练集: 41个（每个区间一个）"
            echo "  - 验证集: 41个（每个区间一个）"
            echo "  - 数据源: /mnt/hdfs/selection/tiktok_cmt"
            echo "  - 配置文件目录: $CONFIG_DIR"
            echo "  - 支持的配置文件: 128k, 256k, 384k, 500k"
            echo ""
            echo "功能特性:"
            echo "  - 自动遍历 $CONFIG_DIR 目录下的所有 YAML 配置文件"
            echo "  - 为每个配置文件生成独立的结果文件 (results_<config_name>.csv)"
            echo "  - 支持多种序列长度配置的并行实验"
            echo ""
            echo "环境变量:"
            echo "  NUM_GPUS         设置GPU数量（默认: 8）"
            echo ""
            echo "示例:"
            echo "  $0                    # 运行所有配置文件的完整实验"
            echo "  $0 --data-only        # 为所有配置文件生成数据"
            echo "  NUM_GPUS=4 $0         # 使用4个GPU运行所有配置的实验"
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            print_info "使用 --help 查看可用选项"
            return 1
            ;;
        esac
        shift
    done
    
    # 如果没有参数，执行主函数
    main_tiktok
}

# 执行参数解析
parse_tiktok_args "$@"