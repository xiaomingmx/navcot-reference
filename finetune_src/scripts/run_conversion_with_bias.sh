#!/bin/bash

# Path to the MODIFIED conversion script that handles biases
CONVERSION_SCRIPT="/root/autodl-tmp/navcot/NavCoT/LLaMA2-Accessory/accessory/tools/convert_weights_to_hf_with_bias.py"

# Path to the source fine-tuned model checkpoint
# This should contain the consolidated.00-of-01.model.pth file
SRC_WEIGHTS_PATH="/root/autodl-tmp/navcot/NavCoT/finetune_src/datasets/R2R/exprs/finetune/agent/finetuned_model-r2r/epoch1"

# Path to the original LLaMA model's params.json config file
SRC_CONFIG_PATH="/root/autodl-tmp/navcot/Data_prepar/Training/tokenizer/code_7B_params.json"

# Path to the tokenizer model file
TOKENIZER_PATH="/root/autodl-tmp/navcot/Data_prepar/Training/tokenizer/tokenizer.model"

# Path where the converted HuggingFace model will be saved
# We add "-with-bias-bf16" to distinguish it and indicate the new precision
DST_WEIGHTS_PATH="/root/autodl-tmp/navcot/NavCoT/finetune_src/datasets/R2R/exprs/finetune/agent/finetuned_model-r2r-hf-with-bias-bf16"

# Data type for the converted weights. Changed to bf16 for better precision preservation.
DTYPE="bf16"

echo "Starting model conversion with bias parameters to bf16 precision..."

python $CONVERSION_SCRIPT \
    --src_weights_path $SRC_WEIGHTS_PATH \
    --src_config_path $SRC_CONFIG_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --dst_weights_path $DST_WEIGHTS_PATH \
    --dtype $DTYPE

echo "Conversion finished. The HuggingFace compatible model (bf16) is saved at: $DST_WEIGHTS_PATH" 