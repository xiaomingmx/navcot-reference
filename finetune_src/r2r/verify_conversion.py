import os
import sys
import json
import torch
import traceback
import torch.distributed as dist
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, AutoTokenizer, AutoModelForCausalLM

# --- Path Setup ---
# Add the LLaMA2-Accessory directory to the Python path to resolve imports
# This makes the script runnable from different locations.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assumes the script is in NavCoT/finetune_src/r2r/ and accessory is in NavCoT/LLaMA2-Accessory/
accessory_path = os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'LLaMA2-Accessory'))
if accessory_path not in sys.path:
    sys.path.insert(0, accessory_path)

# --- Now imports from LLaMA2-Accessory should work ---
from accessory.model.meta import MetaModel
from accessory.util.tensor_parallel import load_tensor_parallel_model_list
from accessory.data.tokenizer import Tokenizer
from accessory.configs.model_args import ModelArgs
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

def load_original_model(model_path, llama_config, tokenizer_path, llama_type):
    """
    Loads the original model using MetaModel.
    """
    # Initialize model parallel world for Fairscale before MetaModel is instantiated.
    # This is crucial because MetaModel's submodules (e.g., ParallelEmbedding)
    # require this to be set up during their __init__.
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size(
        torch.distributed.group.WORLD
    ) == 1:
        initialize_model_parallel(1)

    print("Loading Tokenizer and Original Model Args...")
    tokenizer = Tokenizer(model_path=os.path.join(tokenizer_path, 'tokenizer.model'))
    
    with open(llama_config, "r") as f:
        params = json.load(f)
    
    # These are special args for our fine-tuned model
    model_args = ModelArgs(**params)
    model_args.vocab_size = tokenizer.n_words
    model_args.lora_rank = -1
    model_args.bias_tuning = True
    
    model = MetaModel(
        model_args,
        llama_type=llama_type,
        tokenizer=tokenizer,
        with_visual=False,  # 根据您的模型调整
    )

    # 加载模型权重
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Handle both single-file and sharded checkpoints
    if 'model' in checkpoint:
        model_state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        model_state_dict = checkpoint['state_dict']
    else:
        model_state_dict = checkpoint

    # Need to handle tensor parallel loading if applicable
    load_result = load_tensor_parallel_model_list(model.models, [model_state_dict])
    print(f"Weight loading result: {load_result}")

    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    print("Original model loaded successfully.")
    return model

def load_hf_model(model_path):
    """
    Loads the Hugging Face model and tokenizer.
    """
    print(f"--- Loading Converted HF Model from {model_path} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,  # Use float16 for consistency
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            model.to("cuda")
        print("HF model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"加载 HF 模型失败: {e}")
        print(traceback.format_exc())
        return None, None

def run_comparison(original_model_path, original_config_path, tokenizer_path, hf_model_path, test_prompt):
    """
    Loads both models, generates output for a test prompt, and compares them.
    """
    # --- Initialize a default process group for Fairscale ---
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355' # Or any free port

    # Check if distributed is already initialized
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        print("Initialized default process group.")

    # --- 2. 加载两个模型 ---
    try:
        # ORIGINAL_MODEL_DIR 现在是 weights_path
        # 您可能需要更改 llama_type 以匹配您的模型。
        original_model = load_original_model(
            model_path=original_model_path,
            llama_config=original_config_path,
            tokenizer_path=tokenizer_path,
            llama_type="llama_peft"  # 请在此处指定您的 llama 类型
        )
    except Exception as e:
        print(f"加载原始模型失败，请检查 MetaModel 的初始化方法和路径: {e}")
        print(traceback.format_exc())
        original_model = None

    hf_model, hf_tokenizer = load_hf_model(hf_model_path)

    # --- 3. 准备输入 ---
    # 使用一个在您任务中真实存在的、固定的 prompt
    print(f"\n--- Test Prompt ---\n{test_prompt}\n--------------------")

    # --- 4. 生成输出 ---
    print("\n--- Generating Outputs ---")

    # Generate with original model
    original_output = None
    if original_model:
        try:
            # Note: The original model's generate function might need different parameters.
            # You may need to adjust this call based on the MetaModel.generate implementation.
            # We are assuming it takes a list of strings and returns a list of strings.
            prompts = [test_prompt]
            results = original_model.generate(prompts=prompts, max_gen_len=50, temperature=0, top_p=1.0)
            original_output = results[0]
            print(f"  - Original Model Output: '{original_output.strip()}'")
        except Exception as e:
            print(f"原始模型生成失败: {e}")
            print(traceback.format_exc())
            original_output = f"ERROR: {e}"
    else:
        print("Skipping original model generation as it failed to load.")


    # Generate with HF model
    hf_output = None
    if hf_model and hf_tokenizer:
        try:
            inputs = hf_tokenizer(test_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            outputs = hf_model.generate(**inputs, max_new_tokens=50)
            hf_result_only_generated = hf_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            hf_output = hf_result_only_generated
            print(f"  - HF Model Output:       '{hf_output.strip()}'")
        except Exception as e:
            print(f"HF 模型生成失败: {e}")
            print(traceback.format_exc())
            hf_output = f"ERROR: {e}"
    else:
        print("Skipping HF model generation as it failed to load.")


    # --- 3. 比较结果 ---
    print("\n--- Comparison ---")
    if original_output and hf_output and "ERROR" not in original_output and "ERROR" not in hf_output:
        # A simple string comparison. More sophisticated checks could be implemented.
        are_same = (original_output.strip() == hf_output.strip())
        print(f"Outputs are the same: {are_same}")
        if not are_same:
             print("Warning: Outputs differ.")
    else:
        print("Could not perform comparison because one of the models failed during loading or generation.")


    # Clean up the process group
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Destroyed process group.")

def main():
    # --- 配置路径 ---
    # !! 请务必将这些路径修改为您自己的实际路径 !!
    BASE_PATH = "/root/autodl-tmp/navcot/NavCoT/finetune_src/datasets/R2R/exprs/finetune/agent"
    ORIGINAL_MODEL_PATH = os.path.join(BASE_PATH, "finetuned_model-r2r", "epoch1", "consolidated.00-of-01.model.pth")
    ORIGINAL_CONFIG_PATH = "/root/autodl-tmp/navcot/Data_prepar/Training/tokenizer/code_7B_params.json"
    TOKENIZER_PATH = "/root/autodl-tmp/navcot/Data_prepar/Training/tokenizer"
    HF_MODEL_SAVE_PATH = os.path.join(BASE_PATH, "finetuned_model-r2r-hf")

    run_comparison(
        original_model_path=ORIGINAL_MODEL_PATH,
        original_config_path=ORIGINAL_CONFIG_PATH,
        tokenizer_path=TOKENIZER_PATH,
        hf_model_path=HF_MODEL_SAVE_PATH,
        test_prompt="Translate this sentence to French: Hello, how are you?"
    )

if __name__ == "__main__":
    main()
