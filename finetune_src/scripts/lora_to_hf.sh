python /root/autodl-tmp/navcot/NavCoT/LLaMA2-Accessory/accessory/convert_lora_to_hf.py \
    --llama_type llama_peft \
    --llama_config /root/autodl-tmp/navcot/Data_prepar/Training/tokenizer/code_7B_params.json \
    --tokenizer_path /root/autodl-tmp/navcot/Data_prepar/Training/tokenizer \
    --no_visual \
    --ckpt_path /root/autodl-tmp/navcot/NavCoT/finetune_src/datasets/R2R/exprs/finetune/agent/finetuned_model-r2r/epoch1/consolidated.00-of-01.model.pth \
    --output_dir /root/autodl-tmp/navcot/NavCoT/finetune_src/datasets/R2R/exprs/finetune/agent/finetuned_model-r2r-hf-adapter \
    --hf_base_model meta-llama/Llama-2-7b-hf