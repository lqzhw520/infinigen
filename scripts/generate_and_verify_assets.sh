#!/bin/bash
# generate_and_verify_assets.sh
#
# 一键生成并验证 Infinigen 可动资产
# 包含：生成 MJCF 资产 + 验证关节正确性 + 渲染关节运动视频
#
# 支持的资产类型（18 种）：
#   door, toaster, dishwasher, lamp, cabinet, drawer, refrigerator,
#   oven, microwave, soap_dispenser, faucet, plier, window, box,
#   pepper_grinder, trash, door_handle, stovetop
#
# 使用方法：
#   ./scripts/generate_and_verify_assets.sh [asset_name] [num_seeds] [resolution]
#
# 示例：
#   ./scripts/generate_and_verify_assets.sh cabinet 10 256     # 生成 10 个柜子
#   ./scripts/generate_and_verify_assets.sh drawer 5 512       # 生成 5 个抽屉
#   ./scripts/generate_and_verify_assets.sh all 3 256          # 生成所有 18 种资产，每种 3 个

# 获取脚本所在目录（用于确定相对路径）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 默认参数
ASSET_NAME="${1:-cabinet}"
NUM_SEEDS="${2:-5}"
RESOLUTION="${3:-512}"

# 所有支持的可动资产类型
ALL_ASSETS=(
    "door" "toaster" "dishwasher" "lamp" "cabinet" "drawer"
    "refrigerator" "oven" "microwave" "soap_dispenser" "faucet"
    "plier" "window" "box" "pepper_grinder" "trash" "door_handle" "stovetop"
)

# 固定参数
EXPORTER="mjcf"
OUTPUT_DIR="sim_exports"

#=============================================================================
# 函数：生成并验证单个资产类型
#=============================================================================
generate_single_asset() {
    local asset_name=$1
    local num_seeds=$2
    local resolution=$3
    
    local asset_dir="$OUTPUT_DIR/$EXPORTER/$asset_name"
    local render_dir="${asset_name}_renders"
    local start_seed=1001
    local end_seed=$((start_seed + num_seeds - 1))
    
    echo ""
    echo "============================================================"
    echo "Generating: $asset_name"
    echo "============================================================"
    echo "  Seeds:      $num_seeds ($start_seed-$end_seed)"
    echo "  Asset Dir:  $asset_dir"
    echo "  Render Dir: $render_dir"
    echo "  Resolution: ${resolution}x${resolution}"
    echo ""
    
    # 步骤 1: 生成资产
    echo "[Step 1/2] Generating $num_seeds $asset_name assets..."
    if ! ./scripts/spawn_sim_ready_asset.sh "$asset_name" "$num_seeds" "$EXPORTER"; then
        echo "❌ Failed to generate $asset_name assets"
        return 1
    fi
    echo "[Step 1/2] ✅ Asset generation complete!"
    
    # 步骤 2: 验证并渲染
    echo ""
    echo "[Step 2/2] Verifying and rendering assets..."
    
    # 构建 seeds 参数
    local seeds=""
    for i in $(seq $start_seed $end_seed); do
        seeds="$seeds $i"
    done
    
    if ! python scripts/verify_sim_assets.py \
        --asset_name "$asset_name" \
        --asset_dir "$asset_dir" \
        --seeds $seeds \
        --render \
        --output_dir "$render_dir" \
        --resolution "$resolution"; then
        echo "❌ Failed to verify $asset_name assets"
        return 1
    fi
    
    echo ""
    echo "✅ $asset_name complete!"
    echo "  Assets:  $asset_dir/"
    echo "  Renders: $render_dir/"
    
    return 0
}

#=============================================================================
# 函数：生成所有资产类型
#=============================================================================
generate_all_assets() {
    local num_seeds=$1
    local resolution=$2
    local total_assets=${#ALL_ASSETS[@]}
    local total_instances=$((total_assets * num_seeds))
    
    # 日志文件
    local log_file="sim_assets_generation_$(date +%Y%m%d_%H%M%S).log"
    
    echo "============================================================"
    echo "Infinigen - Generate ALL Articulated Assets"
    echo "============================================================"
    echo "Project Root:    $PROJECT_ROOT"
    echo "Asset Types:     $total_assets types"
    echo "Seeds per Type:  $num_seeds (1001-$((1000 + num_seeds)))"
    echo "Total Instances: $total_instances"
    echo "Resolution:      ${resolution}x${resolution}"
    echo "Log File:        $log_file"
    echo ""
    echo "Asset Types to Generate:"
    for asset in "${ALL_ASSETS[@]}"; do
        echo "  - $asset"
    done
    echo "============================================================"
    echo ""
    
    # 记录开始
    echo "Generation started at $(date)" | tee "$log_file"
    
    # 结果跟踪
    declare -A results
    local success_count=0
    local fail_count=0
    local start_time=$(date +%s)
    
    # 逐个生成
    for i in "${!ALL_ASSETS[@]}"; do
        local asset="${ALL_ASSETS[$i]}"
        local progress=$((i + 1))
        
        echo "" | tee -a "$log_file"
        echo "[$progress/$total_assets] Processing: $asset" | tee -a "$log_file"
        
        if generate_single_asset "$asset" "$num_seeds" "$resolution" 2>&1 | tee -a "$log_file"; then
            results[$asset]="✅ SUCCESS"
            success_count=$((success_count + 1))
        else
            results[$asset]="❌ FAILED"
            fail_count=$((fail_count + 1))
        fi
    done
    
    # 计算时间
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    # 汇总报告
    echo "" | tee -a "$log_file"
    echo "============================================================" | tee -a "$log_file"
    echo "FINAL SUMMARY" | tee -a "$log_file"
    echo "============================================================" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
    echo "Time Elapsed: ${minutes}m ${seconds}s" | tee -a "$log_file"
    echo "Completed at: $(date)" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
    echo "Results by Asset Type:" | tee -a "$log_file"
    echo "------------------------------------------------------------" | tee -a "$log_file"
    for asset in "${ALL_ASSETS[@]}"; do
        echo "  $asset: ${results[$asset]}" | tee -a "$log_file"
    done
    echo "------------------------------------------------------------" | tee -a "$log_file"
    echo "Total: $total_assets types" | tee -a "$log_file"
    echo "  Success: $success_count" | tee -a "$log_file"
    echo "  Failed:  $fail_count" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
    echo "Log saved to: $log_file" | tee -a "$log_file"
    
    # 如果有失败，打印调试信息
    if [ $fail_count -gt 0 ]; then
        echo "" | tee -a "$log_file"
        echo "DEBUG: To retry failed assets, run:" | tee -a "$log_file"
        echo "  ./scripts/generate_and_verify_assets.sh <asset_name> 1 256" | tee -a "$log_file"
    fi
}

#=============================================================================
# 主逻辑
#=============================================================================

# 显示帮助
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: ./scripts/generate_and_verify_assets.sh [asset_name] [num_seeds] [resolution]"
    echo ""
    echo "Arguments:"
    echo "  asset_name  - Asset type to generate (default: cabinet)"
    echo "                Use 'all' to generate all 18 asset types"
    echo "  num_seeds   - Number of seeds to generate (default: 10)"
    echo "  resolution  - Render resolution (default: 256)"
    echo ""
    echo "Supported asset types:"
    echo "  ${ALL_ASSETS[*]}"
    echo ""
    echo "Examples:"
    echo "  ./scripts/generate_and_verify_assets.sh cabinet 10 256"
    echo "  ./scripts/generate_and_verify_assets.sh drawer 5 512"
    echo "  ./scripts/generate_and_verify_assets.sh all 3 256"
    exit 0
fi

# 检查资产类型
if [ "$ASSET_NAME" == "all" ]; then
    # 生成所有资产
    generate_all_assets "$NUM_SEEDS" "$RESOLUTION"
else
    # 检查资产类型是否有效
    valid=false
    for asset in "${ALL_ASSETS[@]}"; do
        if [ "$asset" == "$ASSET_NAME" ]; then
            valid=true
            break
        fi
    done
    
    if [ "$valid" == "false" ]; then
        echo "❌ Error: Invalid asset type '$ASSET_NAME'"
        echo ""
        echo "Valid asset types:"
        echo "  ${ALL_ASSETS[*]}"
        echo "  all (generate all types)"
        exit 1
    fi
    
    echo "============================================================"
    echo "Infinigen Articulated Asset Generation & Verification"
    echo "============================================================"
    echo "Project Root: $PROJECT_ROOT"
    
    # 生成单个资产类型
    if generate_single_asset "$ASSET_NAME" "$NUM_SEEDS" "$RESOLUTION"; then
        echo ""
        echo "============================================================"
        echo "Pipeline Complete!"
        echo "============================================================"
    else
        echo ""
        echo "============================================================"
        echo "Pipeline Failed!"
        echo "============================================================"
        exit 1
    fi
fi
