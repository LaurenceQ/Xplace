#!/bin/bash

# Script to run run_timer.py for all designs

# Training dataset (14 designs)
TRAINING_DESIGNS=(
    "blabla"
    "usb_cdc_core"
    "BM64"
    "salsa20"
    "aes128"
    "wbqspiflash"
    "cic_decimator"
    "aes256"
    "des"
    "aes_cipher"
    "picorv32a"
    "zipdiv"
    "genericfir"
    "usb"
)

# Test dataset (7 designs)
TEST_DESIGNS=(
    "jpeg_encoder"
    "usbf_device"
    "aes192"
    "xtea"
    "spm"
    "y_huff"
    "synth_ram"
)

echo "Running all designs..."
echo ""

# Run training dataset
for design in "${TRAINING_DESIGNS[@]}"; do
    python run_timer.py --designName "$design" 
done

# Run test dataset
for design in "${TEST_DESIGNS[@]}"; do
    python run_timer.py --designName "$design" 
done

echo "Done"