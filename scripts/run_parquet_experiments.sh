#!/bin/bash

# Parquet数据集 SFT Scaling Law 实验自动化脚本
# 该脚本将执行完整的实验流程：parquet数据生成、模型训练和评估
# 
# 分布式训练和评估配置:
# - 使用 torchrun 启动分布式训练和评估，默认使用 8 个 GPU
# - 可通过环境变量 NUM_GPUS 调整 GPU 数量，例如: NUM_GPUS=4 ./run_parquet_experiments.sh
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

# 重写数据生成函数以使用parquet数据构建器
generate_parquet_datasets() {
    print_info "=== 步骤1: 生成Parquet数据集 ==="
    
    if [ -d "data/training" ] && [ -d "data/validation" ] && [ "$(ls -A data/training 2>/dev/null)" ] && [ "$(ls -A data/validation 2>/dev/null)" ]; then
        print_warning "数据集已存在，跳过数据生成步骤"
        print_info "如需重新生成数据集，请删除data目录后重新运行"
        return 0
    fi
    
    print_info "开始生成parquet训练和验证数据集..."
    print_info "使用ParquetDataBuilder处理13个源数据集"
    print_info "将生成53个训练集（1个全量 + 13*4个变体）"
    
    python3 src/parquet_data_builder.py
    
    if [ $? -eq 0 ]; then
        print_success "Parquet数据集生成完成"
        
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
        print_error "Parquet数据集生成失败"
        return 1
    fi
}

# 检查parquet数据源是否可访问
check_parquet_data_source() {
    print_info "检查parquet数据源..."
    
    local base_dir="/mnt/hdfs/selection/from_jiaxiang_wu/general_n_safety_datasets_250925"
    
    if [ ! -d "$base_dir" ]; then
        print_error "parquet数据源目录不存在: $base_dir"
        print_error "请确保HDFS挂载正确或数据源路径正确"
        return 1
    fi
    
    # 检查一些关键文件是否存在
    local test_files=(
        "tulu3_qwen3_2507_no_think_coding_8k.parquet"
        "safety_cn_bias.parquet"
        "open_r1_qwen3_2507_think_coding_8k.parquet"
    )
    
    local found_files=0
    for file in "${test_files[@]}"; do
        if [ -f "$base_dir/$file" ]; then
            found_files=$((found_files + 1))
        fi
    done
    
    if [ $found_files -eq 0 ]; then
        print_error "未找到任何预期的parquet文件"
        print_error "请检查数据源路径和文件是否正确"
        return 1
    fi
    
    print_success "parquet数据源检查通过 (找到 $found_files/${#test_files[@]} 个测试文件)"
    return 0
}

# 检查parquet处理依赖
check_parquet_dependencies() {
    print_info "检查parquet处理依赖..."
    
    # 检查必要的包
    python3 -c "import pandas, pyarrow" 2>/dev/null || {
        print_error "缺少parquet处理依赖，请运行: pip install pandas pyarrow"
        return 1
    }
    
    print_success "parquet依赖检查通过"
    return 0
}

# 主函数 - 专门用于parquet数据集实验
main_parquet() {
    print_info "开始Parquet数据集 SFT Scaling Law实验"
    print_info "实验将包括: parquet数据生成 -> 模型训练 -> 模型评估"
    print_info "预计生成53个训练集和13个验证集"
    
    # 环境检查
    if ! check_python_env; then
        print_error "Python环境检查失败，实验终止"
        return 1
    fi
    
    if ! check_parquet_dependencies; then
        print_error "parquet依赖检查失败，实验终止"
        return 1
    fi
    
    if ! check_parquet_data_source; then
        print_error "parquet数据源检查失败，实验终止"
        return 1
    fi
    
    check_cuda
    check_wandb
    
    # 创建必要的目录
    mkdir -p outputs
    mkdir -p data/training
    mkdir -p data/validation
    
    # 执行实验步骤
    if ! generate_parquet_datasets; then
        print_error "parquet数据生成失败，实验终止"
        return 1
    fi
    
    if ! train_and_evaluate_models; then
        print_error "训练和评估失败，实验终止"
        return 1
    fi
    
    show_results
    
    print_success "Parquet数据集 SFT Scaling Law实验完成!"
    print_info "请查看 results.csv 文件获取详细结果"
    print_info "实验涵盖了13个源数据集的53种训练组合"
    
    # 最终时间统计
    final_end_time=$(date +%s)
    final_total_time=$((final_end_time - SCRIPT_START_TIME))
    print_info "实验总耗时: $(format_time $final_total_time)"
}

# 解析命令行参数 - 扩展原有功能
parse_parquet_args() {
    while [[ $# -gt 0 ]]; do
    case $1 in
        --data-only)
            print_info "仅生成parquet数据集"
            if check_python_env && check_parquet_dependencies && check_parquet_data_source && generate_parquet_datasets; then
                print_success "parquet数据生成完成"
            else
                print_error "parquet数据生成失败"
            fi
            return 0
            ;;
        --train-only)
            print_info "仅执行训练（使用现有parquet数据集）"
            if check_python_env; then
                check_cuda
                check_wandb
                if train_and_evaluate_models; then
                    print_success "训练完成"
                else
                    print_error "训练失败"
                fi
            else
                print_error "Python环境检查失败"
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
            print_info "仅检查parquet数据源"
            if check_parquet_data_source; then
                print_success "数据源检查通过"
            else
                print_error "数据源检查失败"
            fi
            return 0
            ;;
        --help|-h)
            echo "Parquet数据集 SFT Scaling Law 实验脚本"
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --data-only    仅生成parquet数据集"
            echo "  --train-only   仅执行训练"
            echo "  --eval-only    仅执行评估"
            echo "  --check-data   仅检查parquet数据源"
            echo "  --help, -h     显示此帮助信息"
            echo ""
            echo "数据集信息:"
            echo "  - 源数据集: 13个parquet文件"
            echo "  - 训练集: 53个（1个全量 + 13*4个变体）"
            echo "  - 验证集: 13个（每个源数据集一个）"
            echo "  - 数据源: hdfs://harunava/home/byte_pns_pilab_fl/selection/from_jiaxiang_wu/general_n_safety_datasets_250925"
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
    main_parquet
}

# 执行参数解析
parse_parquet_args "$@"