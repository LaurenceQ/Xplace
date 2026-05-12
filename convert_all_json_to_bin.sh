#!/bin/bash

# Script to convert all JSONL timing graph dumps to DGL .bin graphs

DESIGNS=(
    "aes128"
    "aes192"
    "aes256"
    "aes_cipher"
    "blabla"
    "BM64"
    "cic_decimator"
    "des"
    "genericfir"
    "jpeg_encoder"
    "picorv32a"
    "salsa20"
    "spm"
    "synth_ram"
    "techlib"
    "usb"
    "usb_cdc_core"
    "usbf_device"
    "wbqspiflash"
    "xtea"
    "y_huff"
    "zipdiv"
)

# Output directory for .bin files
OUTPUT_DIR="../TimingPredict/graphs"
INPUT_DIR="./output"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Converting all JSONL timing graphs to DGL .bin format..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Total designs: ${#DESIGNS[@]}"
echo ""

CONVERTED=0
FAILED=0

for design in "${DESIGNS[@]}"; do
    JSON_FILE="$INPUT_DIR/${design}.json"
    BIN_FILE="$OUTPUT_DIR/${design}.graph.bin"

    # Check if JSON file exists
    if [ ! -f "$JSON_FILE" ]; then
        echo "⚠ $design: JSON file not found ($JSON_FILE)"
        ((FAILED++))
        continue
    fi

    echo "=========================================="
    echo "Converting: $design"
    echo "  From: $JSON_FILE"
    echo "  To:   $BIN_FILE"
    echo "=========================================="

    python3 ../TimingPredict/load_graph_from_dump.py "$JSON_FILE" "$BIN_FILE" 2>&1 | tail -5

    if [ $? -eq 0 ] && [ -f "$BIN_FILE" ]; then
        FILE_SIZE=$(du -h "$BIN_FILE" | cut -f1)
        echo "✓ $design converted successfully ($FILE_SIZE)"
        ((CONVERTED++))
    else
        echo "✗ $design conversion failed"
        ((FAILED++))
    fi
    echo ""
done

echo "=========================================="
echo "Conversion Summary:"
echo "  Successful: $CONVERTED"
echo "  Failed: $FAILED"
echo "  Total: ${#DESIGNS[@]}"
echo "=========================================="
echo "All graphs saved to: $OUTPUT_DIR"
