#!/bin/bash
#
# Batch Runner for C++ MICROLIFE Simulations
# ==========================================
#
# Usage:
#   ./run_batch_cpp.sh 10    # Run 10 replicates
#
# This script runs multiple C++ simulations with different random seeds
# Perfect for collecting statistical data

set -e  # Exit on error

REPLICATES=${1:-10}  # Default: 10 replicates
CONFIG_FILE=${2:-experiment_simple.cfg}
OUTPUT_BASE="batch_cpp_results"

echo "=========================================="
echo "C++ BATCH RUNNER"
echo "=========================================="
echo "Replicates: $REPLICATES"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_BASE/"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Run simulations
for i in $(seq 1 $REPLICATES); do
    SEED=$((42 + i - 1))  # Seeds: 42, 43, 44, ...
    REP_DIR="${OUTPUT_BASE}/replicate_$(printf '%04d' $i)"

    echo "[$i/$REPLICATES] Running with seed $SEED..."

    mkdir -p "$REP_DIR"

    # Create temporary config with this seed and output dir
    TMP_CONFIG="${REP_DIR}/config.cfg"
    cp "$CONFIG_FILE" "$TMP_CONFIG"

    # Update config (simple sed replacement)
    sed -i "s/^random_seed = .*/random_seed = $SEED/" "$TMP_CONFIG"
    sed -i "s|^output_directory = .*|output_directory = $REP_DIR|" "$TMP_CONFIG"

    # Run simulation
    START_TIME=$(date +%s)
    ./MICROLIFE_ACADEMIC --config "$TMP_CONFIG" > "${REP_DIR}/log.txt" 2>&1
    END_TIME=$(date +%s)

    RUNTIME=$((END_TIME - START_TIME))
    echo "  ✓ Completed in ${RUNTIME}s"

    # Check if data was exported
    if [ -f "${REP_DIR}/population_timeseries.csv" ]; then
        LINES=$(wc -l < "${REP_DIR}/population_timeseries.csv")
        echo "  ✓ Exported $LINES data points"
    else
        echo "  ✗ Warning: No data file generated"
    fi

    echo ""
done

echo "=========================================="
echo "BATCH COMPLETE!"
echo "=========================================="
echo "Total replicates: $REPLICATES"
echo "Output directory: $OUTPUT_BASE/"
echo ""
echo "Next steps:"
echo "  1. Analyze data with Python toolkit:"
echo "     python analysis_toolkit.py --data $OUTPUT_BASE/replicate_0001/population_timeseries.csv"
echo ""
echo "  2. Aggregate all replicates:"
echo "     python aggregate_batch.py --input $OUTPUT_BASE"
echo ""
