#!/bin/bash
# Download NAP pretrained models and data for PhysNAP
# Run this script from a machine with Google Drive access

set -e

PHYSNAP_ROOT="/mnt/afs2/zhuhaowu/infinigen/external/physnap"

mkdir -p "$PHYSNAP_ROOT/data" "$PHYSNAP_ROOT/log"

echo "=== Step 1: Download training data (partnet_mobility_graph_v4 + partnet_mobility_graph_mesh) ==="
echo "Google Drive ID: 1XCE0YadL8LaXCJCkl8TTnYAIlPjIZnsB"
echo "Direct link: https://drive.google.com/file/d/1XCE0YadL8LaXCJCkl8TTnYAIlPjIZnsB/view?usp=sharing"

if [ ! -d "$PHYSNAP_ROOT/data/partnet_mobility_graph_v4" ]; then
    echo "Downloading training data..."
    gdown "1XCE0YadL8LaXCJCkl8TTnYAIlPjIZnsB" -O /tmp/nap_data.zip
    cd "$PHYSNAP_ROOT/data"
    unzip /tmp/nap_data.zip
    rm /tmp/nap_data.zip
    echo "Training data extracted to $PHYSNAP_ROOT/data/"
else
    echo "Training data already exists, skipping."
fi

echo ""
echo "=== Step 2: Download pretrained checkpoint (s1.5_partshape_ae + v6.1_diffusion) ==="
echo "Google Drive ID: 1v5PYwWCevhoRjjn8-Wh5ThVIOr6aouXQ"
echo "Direct link: https://drive.google.com/file/d/1v5PYwWCevhoRjjn8-Wh5ThVIOr6aouXQ/view?usp=sharing"

if [ ! -d "$PHYSNAP_ROOT/log/s1.5_partshape_ae" ]; then
    echo "Downloading pretrained checkpoint..."
    gdown "1v5PYwWCevhoRjjn8-Wh5ThVIOr6aouXQ" -O /tmp/nap_checkpoint.zip
    cd "$PHYSNAP_ROOT/log"
    unzip /tmp/nap_checkpoint.zip
    rm /tmp/nap_checkpoint.zip
    echo "Checkpoint extracted to $PHYSNAP_ROOT/log/"
else
    echo "Checkpoint already exists, skipping."
fi

echo ""
echo "=== Step 3 (Optional): Download PartNet-Mobility raw dataset ==="
echo "Required for conditional generation with point clouds."
echo "Download from: https://sapien.ucsd.edu/downloads"
echo "After downloading, extract to: $PHYSNAP_ROOT/data/partnet-mobility-v0/"

echo ""
echo "=== Verification ==="
echo "Expected directory structure:"
echo "$PHYSNAP_ROOT/"
echo "├── data/"
echo "│   ├── partnet_mobility_graph_mesh/"
echo "│   ├── partnet_mobility_graph_v4/"
echo "│   └── partnet-mobility-v0/  (optional)"
echo "└── log/"
echo "    └── s1.5_partshape_ae/"

ls -la "$PHYSNAP_ROOT/data/" 2>/dev/null || echo "data/ not yet populated"
ls -la "$PHYSNAP_ROOT/log/" 2>/dev/null || echo "log/ not yet populated"

echo ""
echo "=== Done ==="
