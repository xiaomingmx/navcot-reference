import torch
import json
import os
from collections import defaultdict
import argparse
import torch.distributed as dist
from fairscale.nn.model_parallel import initialize as fs_init
import tempfile
from model.meta import MetaModel 
from util.tensor_type import default_tensor_type
from ..configs.global_configs import get_model_config, ModelArgs
from model.LLM.llama_peft import Transformer as PeftTransformer

def setup_distributed_env():
    """Initializes a fake distributed environment for single-process model loading."""
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')  # Use a free port

    # Using 'nccl' as the backend for GPU support. Use 'gloo' for CPU.
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    # Initialize Fairscale model parallel services.
    # For single-process loading, model parallel size is 1.
    fs_init.initialize_model_parallel(1)
    
    # Set the current device.
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get('RANK', 0)))

def main(args):
    # Set up the distributed environment before any model initialization
    setup_distributed_env()

    # --- 关键修复：创建一个临时的、修正了 lora_rank 的配置文件 ---
    temp_config_path = None
    try:
        # 1. 读取原始配置文件
        with open(args.llama_config, 'r') as f:
            config_data = json.load(f)

        # 2. 使用命令行参数覆盖 lora_rank
        print(f"Overriding lora_rank in config. Original: {config_data.get('lora_rank', 'Not Set')}, New: {args.lora_rank}")
        config_data['lora_rank'] = args.lora_rank

        # 3. 写入临时文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_f:
            json.dump(config_data, temp_f)
            temp_config_path = temp_f.name
        
        print(f"Using temporary config file: {temp_config_path}")

        # 1. 加载您的模型
        # 现在使用临时的配置文件路径
        llama_config = [temp_config_path]
        
        print("Initializing model with corrected config...")
        with default_tensor_type(dtype=torch.bfloat16, device="cpu"):
            model = MetaModel(
                llama_type=args.llama_type, 
                llama_config=llama_config,
                tokenizer_path=args.tokenizer_path, 
                with_visual=not args.no_visual
            )
        
        print(f"Loading checkpoint from: {args.ckpt_path}")
        # 加载您训练好的 checkpoint
        # 注意：这里需要加载整个模型文件，而不是 state_dict
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint['model'], strict=False)
        
        model.eval()
        print("Model loaded successfully.")

        # 2. 手动提取 LoRA 和基础权重
        lora_A_weights = {}
        lora_B_weights = {}
        base_weights = {}

        for name, module in model.named_modules():
            if 'Lora' in module.__class__.__name__:
                # 找到 LoRA 层
                lora_A = getattr(module, 'lora_a', None)
                lora_B = getattr(module, 'lora_b', None)
                
                if lora_A is not None and lora_B is not None:
                    # 转换为HF格式的名称
                    # 1. 移除模型包装前缀，并添加HF PEFT格式要求的前缀
                    hf_name = name.replace('llma.', 'base_model.model.')

                    # 2. 映射内部模块名称到HuggingFace标准名称
                    name_mapping = {
                        '.wq': '.q_proj', '.wk': '.k_proj', '.wv': '.v_proj', '.wo': '.o_proj',
                        '.w1': '.gate_proj', '.w3': '.up_proj', '.w2': '.down_proj'
                    }
                    
                    original_hf_name = hf_name
                    for old_suffix, new_suffix in name_mapping.items():
                        if hf_name.endswith(old_suffix):
                            hf_name = hf_name[:-len(old_suffix)] + new_suffix
                            break
                    
                    if hf_name == original_hf_name:
                        print(f"警告: 模块 '{name}' 的HF名称映射未成功，跳过此模块。")
                        continue

                    # 提取 LoRA 权重
                    lora_A_weights[f"{hf_name}.lora_A.weight"] = lora_A.weight.data
                    lora_B_weights[f"{hf_name}.lora_B.weight"] = lora_B.weight.data
        
        if not lora_A_weights or not lora_B_weights:
            print("错误：未能从模型中提取出任何 LoRA A/B 权重！请检查模型结构和脚本中的名称映射。")
            return

        print(f"成功提取了 {len(lora_A_weights)} 个 LoRA A 权重和 {len(lora_B_weights)} 个 LoRA B 权重。")

        # 3. 计算增量并合并
        merged_weights = {}
        for key_A in lora_A_weights.keys():
            key_B = key_A.replace('lora_A.weight', 'lora_B.weight')
            if key_B in lora_B_weights:
                delta_w = lora_B_weights[key_B] @ lora_A_weights[key_A]
                
                # 找到对应的基础权重并合并
                base_key = key_A.replace('.lora_A.weight', '.weight')
                # 假设基础模型已经加载，或者我们需要从另一个文件加载
                # 在这个脚本中，我们直接创建一个新的 state_dict
                merged_weights[base_key] = delta_w # 这里只保存了增量
                
        # 完整的做法是加载原始 LLaMA 权重，然后加上 delta_w
        # 为了简化，我们这里只保存adapter部分
        
        adapter_state_dict = {}
        for key, value in lora_A_weights.items():
            adapter_state_dict[key] = value
        for key, value in lora_B_weights.items():
            adapter_state_dict[key] = value

        # 4. 保存为 HuggingFace PEFT 格式
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存 adapter 权重
        adapter_save_path = os.path.join(output_dir, 'adapter_model.bin')
        torch.save(adapter_state_dict, adapter_save_path)
        print(f"Adapter weights saved to {adapter_save_path}")

        # 创建并保存 adapter_config.json
        adapter_config = {
            "base_model_name_or_path": args.hf_base_model,
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 16,  # LoRA rank, 需要根据您的配置修改
            "lora_alpha": 32, # 需要根据您的配置修改
            "lora_dropout": 0.05, # 需要根据您的配置修改
            "target_modules": [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "bias": "none",
        }
        config_save_path = os.path.join(output_dir, 'adapter_config.json')
        with open(config_save_path, 'w') as f:
            json.dump(adapter_config, f, indent=4)
        print(f"Adapter config saved to {config_save_path}")

        print("\\n转换完成！")
        print(f"您现在可以使用以下方式加载模型：")
        print("from peft import PeftModel")
        print("from transformers import AutoModelForCausalLM")
        print(f"base_model = AutoModelForCausalLM.from_pretrained('{args.hf_base_model}')")
        print(f"peft_model = PeftModel.from_pretrained(base_model, '{output_dir}')")

    finally:
        # 4. 清理临时文件
        if temp_config_path and os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            print(f"Cleaned up temporary config file: {temp_config_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert NavCoT LoRA checkpoint to HuggingFace PEFT format.")
    parser.add_argument('--llama_type', type=str, default="llama_peft", help='Type of llama model (e.g., llama_peft).')
    parser.add_argument('--llama_config', type=str, required=True, help='Path to llama model config json.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer model file.')
    parser.add_argument('--no_visual', action='store_true', help='Do not initialize visual modules.')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the trained NavCoT checkpoint file (.pth).')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the HuggingFace PEFT adapter.')
    parser.add_argument('--hf_base_model', type=str, required=True, help='HuggingFace model name or path for the base LLaMA model (e.g., "meta-llama/Llama-2-7b-hf").')
    parser.add_argument('--lora_rank', type=int, default=16, help='The rank of the LoRA layers used during training.')

    args = parser.parse_args()
    main(args) 