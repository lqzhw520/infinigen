#!/bin/bash
# Train PhysNAP with Infinigen data
# This script provides the full pipeline from data preparation to training.
#
# Prerequisites:
# 1. PhysNAP environment set up: conda activate physnap
# 2. NAP pretrained shape AE downloaded: external/physnap/log/s1.5_partshape_ae/
# 3. Infinigen data converted: external/physnap/data/infinigen_graph_combined/
#
# Usage:
#   bash scripts/train_physnap_infinigen.sh [option_a|option_b|option_c]

set -e

INFINIGEN_ROOT="/mnt/afs2/zhuhaowu/infinigen"
PHYSNAP_ROOT="$INFINIGEN_ROOT/external/physnap"
DATA_DIR="$PHYSNAP_ROOT/data/infinigen_graph_combined"

MODE=${1:-option_b}

echo "=== PhysNAP Training with Infinigen Data ==="
echo "Mode: $MODE"
echo ""

# Verify prerequisites
if [ ! -d "$PHYSNAP_ROOT/log/s1.5_partshape_ae" ]; then
    echo "ERROR: NAP shape AE not found at $PHYSNAP_ROOT/log/s1.5_partshape_ae/"
    echo "Please run: bash scripts/download_nap_data.sh"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Infinigen NAP data not found at $DATA_DIR"
    echo "Please run the conversion pipeline first."
    exit 1
fi

case $MODE in
    option_a)
        echo "=== Option A: Mixed Training (PartNet-Mobility + Infinigen) ==="
        echo "Merge PartNet-Mobility and Infinigen datasets, then train."

        if [ ! -d "$PHYSNAP_ROOT/data/partnet_mobility_graph_v4" ]; then
            echo "ERROR: PartNet-Mobility data not found."
            echo "Please download it first via scripts/download_nap_data.sh"
            exit 1
        fi

        MERGED_DIR="$PHYSNAP_ROOT/data/mixed_partnet_infinigen"
        mkdir -p "$MERGED_DIR"

        cp "$PHYSNAP_ROOT/data/partnet_mobility_graph_v4/"*.npz "$MERGED_DIR/" 2>/dev/null || true
        cp "$DATA_DIR/"*.npz "$MERGED_DIR/" 2>/dev/null || true

        echo "Merged dataset: $(ls "$MERGED_DIR/"*.npz | wc -l) total samples"

        cd "$PHYSNAP_ROOT"
        python run.py --config ./configs/nap/v6.1_diffusion_adapted.yaml -f \
            --override dataset.data_root="$MERGED_DIR" \
            --override dataset.split_path="$MERGED_DIR/mixed_split.json" \
            --override dataset.embedding_index_file="$MERGED_DIR/mixed_partkeys.json"
        ;;

    option_b)
        echo "=== Option B: Infinigen-Only Training ==="
        echo "Train NAP solely on Infinigen box data (3 types)."
        echo "Data dir: $DATA_DIR"
        echo "Samples: $(ls "$DATA_DIR/"*.npz | wc -l)"

        cd "$PHYSNAP_ROOT"

        cat > /tmp/infinigen_train_config.yaml << 'YAML_EOF'
# PhysNAP training config for Infinigen-only data
# Based on v6.1_diffusion_adapted.yaml with adjusted paths

inherit: ./configs/nap/v6.1_diffusion_adapted.yaml

dataset:
  data_root: PLACEHOLDER_DATA_ROOT
  split_path: PLACEHOLDER_SPLIT
  embedding_precompute_path: PLACEHOLDER_CODEBOOK
  embedding_index_file: PLACEHOLDER_PARTKEYS
  max_K: 8
  cates: ["all"]
  scale_all: True
  scale_mode: max
YAML_EOF

        # Replace placeholders
        sed -i "s|PLACEHOLDER_DATA_ROOT|$DATA_DIR|g" /tmp/infinigen_train_config.yaml
        sed -i "s|PLACEHOLDER_SPLIT|$DATA_DIR/infinigen_split.json|g" /tmp/infinigen_train_config.yaml
        sed -i "s|PLACEHOLDER_CODEBOOK|$DATA_DIR/infinigen_codebook.npz|g" /tmp/infinigen_train_config.yaml
        sed -i "s|PLACEHOLDER_PARTKEYS|$DATA_DIR/infinigen_partkeys.json|g" /tmp/infinigen_train_config.yaml

        echo "Generated training config at /tmp/infinigen_train_config.yaml"
        echo "Starting training..."

        python run.py --config /tmp/infinigen_train_config.yaml -f
        ;;

    option_c)
        echo "=== Option C: Fine-tune from PartNet-Mobility pretrained ==="
        echo "Start from PartNet-Mobility weights, fine-tune on Infinigen."

        if [ ! -d "$PHYSNAP_ROOT/log/v6.1_diffusion_adapted" ]; then
            echo "ERROR: Pretrained NAP diffusion model not found."
            echo "Please train on PartNet-Mobility first (Part 2.3)."
            exit 1
        fi

        cd "$PHYSNAP_ROOT"
        echo "Fine-tuning not yet implemented. Train from scratch first (option_b)."
        ;;

    *)
        echo "Usage: $0 [option_a|option_b|option_c]"
        echo "  option_a: Mixed PartNet + Infinigen training"
        echo "  option_b: Infinigen-only training (default)"
        echo "  option_c: Fine-tune from PartNet pretrained"
        exit 1
        ;;
esac

echo ""
echo "=== Training complete ==="
echo "Check logs at: $PHYSNAP_ROOT/log/"
