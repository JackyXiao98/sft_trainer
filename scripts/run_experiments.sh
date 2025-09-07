#!/bin/bash

# SFT Scaling Law 实验自动化脚本
# 该脚本将执行完整的实验流程：数据生成、模型训练和评估
# 
# 分布式训练和评估配置:
# - 使用 torchrun 启动分布式训练和评估，默认使用 8 个 GPU
# - 可通过环境变量 NUM_GPUS 调整 GPU 数量，例如: NUM_GPUS=4 ./run_experiments.sh
# - 确保系统有足够的 GPU 资源
#
# 安全特性:
# - 脚本错误不会关闭你的terminal
# - 使用函数返回值而非exit进行错误处理
# - 支持优雅的错误恢复

# 注意：不使用 set -e 以避免关闭terminal，改用手动错误检查

# 导入通用函数库
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_functions.sh"

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
        return 0
    else
        print_error "数据集生成失败"
        return 1
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

# 主函数
main() {
    print_info "开始SFT Scaling Law实验"
    print_info "实验将包括: 数据生成 -> 模型训练 -> 模型评估"
    
    # 环境检查
    # if ! check_python_env; then
    #     print_error "Python环境检查失败，实验终止"
    #     return 1
    # fi
    check_cuda
    check_wandb
    
    # 创建必要的目录
    mkdir -p outputs
    mkdir -p data/training
    mkdir -p data/validation
    
    # 执行实验步骤
    if ! generate_datasets; then
        print_error "数据生成失败，实验终止"
        return 1
    fi
    
    if ! train_and_evaluate_models; then
        print_error "训练和评估失败，实验终止"
        return 1
    fi
    
    show_results
    
    print_success "SFT Scaling Law实验完成!"
    print_info "请查看 results.csv 文件获取详细结果"
    print_info "训练好的模型保存在 outputs/ 目录下"
    
    # 最终时间统计
    final_end_time=$(date +%s)
    final_total_time=$((final_end_time - SCRIPT_START_TIME))
    print_info "实验总耗时: $(format_time $final_total_time)"
}

# 信号处理
trap cleanup EXIT

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
    case $1 in
        --data-only)
            print_info "仅生成数据集"
            if check_python_env && generate_datasets; then
                print_success "数据生成完成"
            else
                print_error "数据生成失败"
            fi
            return 0
            ;;
        --train-only)
            print_info "仅执行训练"
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
                # 这里需要实现单独的评估逻辑
                print_warning "单独评估功能需要进一步实现"
                show_results
            else
                print_error "Python环境检查失败"
            fi
            return 0
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
            return 1
            ;;
        esac
        shift
    done
    
    # 如果没有参数，执行主函数
    main
}

# 执行参数解析
parse_args "$@"