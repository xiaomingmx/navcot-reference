python /root/autodl-tmp/navcot/NavCoT/LLaMA2-Accessory/accessory/tools/convert_weights_to_hf.py \
    --src_weights_path /root/autodl-tmp/navcot/NavCoT/finetune_src/datasets/R2R/exprs/finetune/agent/finetuned_model-r2r/epoch1 \
    --src_config_path /root/autodl-tmp/navcot/Data_prepar/Training/tokenizer/code_7B_params.json \
    --tokenizer_path /root/autodl-tmp/navcot/Data_prepar/Training/tokenizer/tokenizer.model \
    --dst_weights_path /root/autodl-tmp/navcot/NavCoT/finetune_src/datasets/R2R/exprs/finetune/agent/finetuned_model-r2r-hf \
    --dtype fp16

