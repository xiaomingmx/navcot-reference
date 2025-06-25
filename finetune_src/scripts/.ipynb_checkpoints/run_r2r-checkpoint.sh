ob_type=cand
feedback=sample 
# feedback=teacher

features=vit-16-ori
ft_dim=512

ngpus=1
seed=0

outdir=/root/autodl-tmp/navcot/NavCoT/finetune_src/datasets/R2R/exprs/finetune/agent/

flag="--root_dir /root/autodl-tmp/navcot/NavCoT/finetune_src/datasets
      --output_dir ${outdir}

      --dataset r2r

      --vlnbert ${vlnbert}
      --ob_type ${ob_type}

      --world_size ${ngpus}
      --seed ${seed}

      --num_l_layers 9
      --num_x_layers 4

      --hist_enc_pano
      --hist_pano_num_layers 2

      --fix_lang_embedding
      --fix_hist_embedding

      --features ${features}
      --feedback ${feedback}

      --max_action_len 15
      --max_instr_len 60

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 300000
      --log_every 2000
      --batch_size 32
      --optim adamW

      --ml_weight 0.1

      --feat_dropout 0.4
      --dropout 0.5"

# inference
cd /root/autodl-tmp/navcot/NavCoT/finetune_src
export PYTHONPATH=/root/autodl-tmp/navcot/NavCoT/finetune_src:$PYTHONPATH
CUDA_VISIBLE_DEVICES='0' torchrun --master_port 12229 --nproc_per_node 1 r2r/main.py $flag  \
      --llm_predict --stop_first --pretrained_path /root/autodl-tmp/navcot/NavCoT/finetune_src/datasets/R2R/exprs/finetune/agent/finetuned_model-r2r/epoch1 \
      --llama_type llama_peft --llama_config='/root/autodl-tmp/navcot/Data_prepar/Training/tokenizer/code_7B_params.json' --no_visual --pretrained_type consolidated \
      --tokenizer_path /root/autodl-tmp/navcot/Data_prepar/Training/tokenizer  --dtype fp16 \
      --test --submit \
