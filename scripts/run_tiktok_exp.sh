#!/bin/bash

# TikTok评论数据集 SFT Scaling Law 实验自动化脚本
# 该脚本将执行完整的实验流程：TikTok数据生成、模型训练和评估
# 
# 分布式训练和评估配置:
# - 使用 torchrun 启动分布式训练和评估，默认使用 8 个 GPU
# - 可通过环境变量 NUM_GPUS 调整 GPU 数量，例如: NUM_GPUS=4 ./run_tiktok_exp.sh
# - 确保系统有足够的 GPU 资源
#
# 安全特性:
# - 脚本错误不会关闭你的terminal
# - 使用函数返回值而非exit进行错误处理
# - 支持优雅的错误恢复

# 注意：不使用 set -e 以避免关闭terminal，改用手动错误检查

# 导入通用函数库（只包含函数定义，不会执行任何主逻辑）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_functions.sh"

# 重写数据生成函数以使用TikTok数据构建器
generate_tiktok_datasets() {
    print_info "=== 步骤1: 生成TikTok评论数据集 ==="
    
    if [ -d "data/training" ] && [ -d "data/validation" ] && [ "$(ls -A data/training 2>/dev/null)" ] && [ "$(ls -A data/validation 2>/dev/null)" ]; then
        print_warning "数据集已存在，跳过数据生成步骤"
        print_info "如需重新生成数据集，请删除data目录后重新运行"
        return 0
    fi
    
    print_info "开始生成TikTok评论训练和验证数据集..."
    print_info "使用TikTokCommentDataBuilder处理41个区间数据集"
    print_info "将生成41个训练集和41个验证集"
    
    # 使用 training_config_thoth.yaml 配置文件
    python3 src/data_split/tiktok_comment_data_builder.py --config configs/training_config_thoth.yaml
    
    if [ $? -eq 0 ]; then
        print_success "TikTok评论数据集生成完成"
        
        # 显示生成的数据集统计
        if [ -d "data/training" ]; then
            training_count=$(ls data/training/ | wc -l)
            print_info "生成的训练数据集数量: $training_count"
        fi
        
        if [ -d "data/validation" ]; then
            validation_count=$(ls data/validation/ | wc -l)
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
    print_info "检查TikTok实验配置文件..."
    
    local config_file="configs/training_config_thoth.yaml"
    
    if [ ! -f "$config_file" ]; then
        print_error "配置文件不存在: $config_file"
        return 1
    fi
    
    # 检查配置文件中的关键配置
    if ! grep -q "model_name.*Qwen3-8B" "$config_file"; then
        print_warning "配置文件中未找到Qwen3-8B模型配置"
    fi
    
    if ! grep -q "use_flash_attention_2.*true" "$config_file"; then
        print_warning "配置文件中未启用flash_attention_2"
    fi
    
    print_success "TikTok配置文件检查通过"
    return 0
}

# 重写训练和评估函数以使用thoth配置
train_and_evaluate_tiktok_models() {
    print_info "=== 步骤2: 训练和评估TikTok模型（使用thoth配置） ==="
    
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
        
        # 使用torchrun启动分布式训练，指定thoth配置文件
        torchrun --nproc_per_node=$NUM_GPUS \
                 --master_port=17238 \
                 src/train.py \
            --dataset_path "data/training/$dataset" \
            --output_dir "$output_dir" \
            --config "configs/training_config_thoth.yaml" \
            --run_name "tiktok-sft-$dataset"
        
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
    
    print_success "所有TikTok模型训练和评估完成"
}

# 主函数 - 专门用于TikTok数据集实验
main_tiktok() {
    print_info "开始TikTok评论数据集 SFT Scaling Law实验"
    print_info "实验将包括: TikTok数据生成 -> 模型训练 -> 模型评估"
    print_info "预计生成41个训练集和41个验证集"
    print_info "使用配置文件: configs/training_config_thoth.yaml"
    
    # 环境检查
    if ! check_python_env; then
        print_error "Python环境检查失败，实验终止"
        return 1
    fi
    
    if ! check_tiktok_dependencies; then
        print_error "TikTok依赖检查失败，实验终止"
        return 1
    fi
    
    if ! check_tiktok_config; then
        print_error "TikTok配置检查失败，实验终止"
        return 1
    fi
    
    if ! check_tiktok_data_source; then
        print_error "TikTok数据源检查失败，实验终止"
        return 1
    fi
    
    check_cuda
    check_wandb
    
    # 创建必要的目录
    mkdir -p outputs
    mkdir -p data/training
    mkdir -p data/validation
    
    # 执行实验步骤
    if ! generate_tiktok_datasets; then
        print_error "TikTok数据生成失败，实验终止"
        return 1
    fi
    
    if ! train_and_evaluate_tiktok_models; then
        print_error "训练和评估失败，实验终止"
        return 1
    fi
    
    show_results
    
    print_success "TikTok评论数据集 SFT Scaling Law实验完成!"
    print_info "请查看 results.csv 文件获取详细结果"
    print_info "实验涵盖了41个TikTok区间数据集的训练组合"
    
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
            print_info "仅生成TikTok评论数据集"
            if check_python_env && check_tiktok_dependencies && check_tiktok_config && check_tiktok_data_source && generate_tiktok_datasets; then
                print_success "TikTok数据生成完成"
            else
                print_error "TikTok数据生成失败"
            fi
            return 0
            ;;
        --train-only)
            print_info "仅执行训练（使用现有TikTok数据集）"
            if check_python_env && check_tiktok_config; then
                check_cuda
                check_wandb
                if train_and_evaluate_tiktok_models; then
                    print_success "训练完成"
                else
                    print_error "训练失败"
                fi
            else
                print_error "环境检查失败"
            fi
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
            print_info "仅检查配置文件"
            if check_tiktok_config; then
                print_success "配置文件检查通过"
            else
                print_error "配置文件检查失败"
            fi
            return 0
            ;;
        --help|-h)
            echo "TikTok评论数据集 SFT Scaling Law 实验脚本"
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
            echo "  - 配置文件: configs/training_config_thoth.yaml"
            echo ""
            echo "环境变量:"
            echo "  NUM_GPUS         设置GPU数量（默认: 8）"
            echo ""
            echo "示例:"
            echo "  $0                    # 运行完整实验"
            echo "  $0 --data-only        # 仅生成数据"
            echo "  NUM_GPUS=4 $0         # 使用4个GPU运行实验"
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