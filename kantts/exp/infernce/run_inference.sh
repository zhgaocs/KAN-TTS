#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

INPUT_FILE="../input/example_sentences.txt"
OUTPUT_DIR="output/$(date +%Y%m%d_%H%M%S)"
RESOURCE_ZIP="../models/speech_sambert-hifigan_tts_chuangirl_Sichuan_16k/resource.zip"
VOC_CKPT="../models/speech_sambert-hifigan_tts_chuangirl_Sichuan_16k/basemodel_16k/hifigan/ckpt/checkpoint_340000.pth"
AM_CKPT="../models/speech_sambert-hifigan_tts_chuangirl_Sichuan_16k/basemodel_16k/sambert/ckpt/checkpoint_980000.pth"

mkdir -p "$OUTPUT_DIR"

python ../../bin/text_to_wav.py \
    --txt "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --res_zip "$RESOURCE_ZIP" \
    --voc_ckpt "$VOC_CKPT" \
    --am_ckpt "$AM_CKPT"

echo "Inference completed! Results are saved to $OUTPUT_DIR"
