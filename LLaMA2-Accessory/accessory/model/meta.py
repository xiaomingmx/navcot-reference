import torch
import torch.nn as nn
import json
from typing import List, Optional, Tuple
import heapq

from fairscale.nn.model_parallel import initialize as fs_init

from .tokenizer import Tokenizer
from . import LLM
from util import misc

# Try to import vLLM, handle if not installed
try:
    from vllm import LLM as VLLM_Engine, SamplingParams as VLLM_SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    VLLM_Engine = None
    VLLM_SamplingParams = None
    print("Warning: vLLM not found. `generate_with_vllm` will not be available.")


class MetaModel(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(
        self, 
        llama_type: Optional[str] = None, # Made optional if use_vllm is True
        llama_config: Optional[List[str]] = None, # Made optional
        tokenizer_path: Optional[str] = None, # Made optional
        with_visual: bool = False, 
        max_seq_len: int = 2048,
        use_vllm: bool = False,
        vllm_hf_model_path: Optional[str] = None,
        vllm_args: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.use_vllm = use_vllm
        self.weights_loaded_by_vllm = False
        self.vllm_engine_instance = None
        self.llma = None # Ensure llma is initialized to None

        if self.use_vllm:
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM is selected but not available. Please install vLLM.")
            if not vllm_hf_model_path:
                raise ValueError("vLLM is selected but vllm_hf_model_path is not provided.")
            
            assert VLLM_Engine is not None, "VLLM_Engine is None despite VLLM_AVAILABLE being True. This should not happen."
            assert VLLM_SamplingParams is not None, "VLLM_SamplingParams is None despite VLLM_AVAILABLE being True. This should not happen."

            print(f"Initializing vLLM engine with model: {vllm_hf_model_path}")
            vllm_params = vllm_args if vllm_args is not None else {}
            
            # Default tensor_parallel_size to 1 if not in vllm_args
            tp_size = vllm_params.get("tensor_parallel_size", 1)
            
            # Ensure tokenizer path for VLLM engine is set, can be same as model path
            # vLLM typically finds tokenizer in the model directory.
            vllm_tokenizer_for_engine = vllm_params.get("tokenizer", vllm_hf_model_path)

            # Get and normalize dtype for vLLM
            vllm_engine_dtype_str = vllm_params.get("dtype", "auto")
            if isinstance(vllm_engine_dtype_str, str):
                if vllm_engine_dtype_str.lower() == "fp16":
                    print(f"Info: Converting dtype 'fp16' to 'float16' for vLLM engine.")
                    vllm_engine_dtype_str = "float16"
                elif vllm_engine_dtype_str.lower() == "bf16": # Ensure bf16 is also passed as string if needed
                    vllm_engine_dtype_str = "bfloat16"
            # If vllm_engine_dtype_str is already a torch.dtype or "auto", it should be fine.

            # 检查是否在分布式环境中，如果是则强制使用单GPU模式避免冲突
            import torch.distributed as dist
            if dist.is_initialized():
                print("[21:06:18] Warning: Distributed training detected. Setting vLLM to single GPU mode to avoid tensor parallel conflicts.")
                tp_size = 1  # 强制使用单GPU
                # 添加 enforce_eager 参数避免 CUDA graphs 在分布式环境中的问题
                vllm_params["enforce_eager"] = True
            
            self.vllm_engine_instance = VLLM_Engine(
                model=vllm_hf_model_path,
                tokenizer=vllm_tokenizer_for_engine, 
                tensor_parallel_size=tp_size,
                dtype=vllm_engine_dtype_str, # Use the potentially corrected dtype string
                max_model_len=max_seq_len,  # Add max_seq_len parameter for vLLM
                # Pass other args from vllm_args, filtering out those already handled
                **{k: v for k, v in vllm_params.items() if k not in ["tensor_parallel_size", "tokenizer", "dtype", "max_model_len"]}
            )
            self.tokenizer = self.vllm_engine_instance.get_tokenizer()
            self.weights_loaded_by_vllm = True
            print("vLLM engine initialized and weights loaded.")
            # For vLLM, we don't initialize self.llma or the original LLaMA components
            # However, criterion might still be needed if training is done elsewhere or on a base model part
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # Assuming 0 is pad_id for tokenizer
            # If tokenizer has pad_id, use it for ignore_index
            # For vLLM, self.tokenizer is obtained from vllm_engine.get_tokenizer()
            # We need to check if this vLLM tokenizer wrapper has pad_id.
            vllm_tokenizer_pad_id = getattr(self.tokenizer, 'pad_token_id', getattr(self.tokenizer, 'pad_id', None))
            if vllm_tokenizer_pad_id is not None:
                 self.criterion = torch.nn.CrossEntropyLoss(ignore_index=vllm_tokenizer_pad_id)
                 print(f"vLLM mode: Criterion using pad_id {vllm_tokenizer_pad_id} from vLLM tokenizer.")
            # else: # Fallback or if pad_id is 0 or not explicitly set in vLLM's tokenizer wrapper
            #      # We need to know the pad_id. If it's from the HF tokenizer, it should be there.
            #      # Defaulting to 0 if not found, but this might need adjustment.
            #      # It's better if vLLM's tokenizer wrapper exposes pad_id correctly.
            #      # For now, if tokenizer_path was provided, try loading a separate tokenizer
            #      # instance just to get pad_id if needed. This is a bit of a workaround.
            #      temp_hf_tokenizer = None
            #      if tokenizer_path: # The original tokenizer_path for HF
            #          try:
            #              from transformers import AutoTokenizer
            #              temp_hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            #              if temp_hf_tokenizer.pad_token_id is not None:
            #                 self.criterion = torch.nn.CrossEntropyLoss(ignore_index=temp_hf_tokenizer.pad_token_id)
            #                 print(f"Using pad_id {temp_hf_tokenizer.pad_token_id} for criterion from external tokenizer {tokenizer_path}")
            #              else:
            #                 print(f"Warning: External tokenizer {tokenizer_path} has no pad_token_id. Criterion using ignore_index=0.")
            #          except Exception as e:
            #              print(f"Warning: Could not load external tokenizer from {tokenizer_path} to get pad_id: {e}. Criterion using ignore_index=0.")
            #      else: # If original tokenizer_path is also not given
            #          print("Warning: pad_id for criterion with vLLM is ambiguous. Defaulting to ignore_index=0. Ensure this is correct.")


        else: # Not using vLLM, proceed with original LLaMA setup

            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

            ModelArgs = LLM.__dict__[llama_type].ModelArgs
            Transformer = LLM.__dict__[llama_type].Transformer


            params = {}
            for _ in llama_config:
                with open(_, "r") as f:
                    params.update(json.loads(f.read()))

            model_args: ModelArgs = ModelArgs(
                max_seq_len=max_seq_len, max_batch_size=32, **params
            )

            self.tokenizer = Tokenizer(model_path=tokenizer_path)
            model_args.vocab_size = self.tokenizer.n_words
            

            print("Model Args:\n", model_args)

            # Create model only in non-vLLM mode
            model = Transformer(model_args, with_visual=with_visual)
            self.llma = model

            # Common initialization for both vLLM and non-vLLM modes
            if not self.use_vllm:
                self._set_default_trainability()

            # Only set model properties if we have a model (non-vLLM mode)
            if hasattr(self, 'llma') and self.llma is not None:
                self.is_peft = getattr(self.llma, "is_peft", False)
            print(f"Model is Peft: {self.is_peft}")

            misc.mark_mp_params(self)

            param_count_local, param_count_all = 0, 0
            for name, param in self.named_parameters():
                is_model_parallel = getattr(param, "is_model_parallel", False)
                if param.requires_grad:
                    if is_model_parallel:
                        param_count_all += param.numel() * fs_init.get_model_parallel_world_size()
                    else:
                        param_count_all += param.numel()
                    param_count_local += param.numel()
            print(f"Trainable parameter count : {param_count_local} (local rank), {param_count_all} (all).")
                
            # Only set this flag for non-vLLM mode
            if not use_vllm:
                self.weights_loaded_by_vllm = False


    def get_trainable_params(self):
        if self.use_vllm:
            # If using vLLM, trainable parameters are not managed by this class for 'llma'
            # This method might need re-evaluation depending on fine-tuning strategy with vLLM
            # (e.g., if fine-tuning adapters on top of a vLLM-served base model,
            #  this would need to target those adapter parameters).
            # For now, returning empty or raising an error might be appropriate if not fine-tuning.
            print("Warning: get_trainable_params() called with use_vllm=True. Fine-tuning with vLLM engine directly is not standard.")
            return {} 
        if self.llma is None: # Should not happen if not use_vllm, but as a safeguard
             return {}
        llma_trainable = self.llma.get_trainable_params()
        return {"llma." + name: param for name, param in llma_trainable.items()}


    def _set_default_trainability(self):
        if self.use_vllm:
            # No equivalent operation for vLLM engine from here. Weights are managed by vLLM.
            # This function primarily sets requires_grad for self.llma parameters.
            print("Info: _set_default_trainability() skipped as use_vllm is True.")
            return

        for key, value in self.named_parameters():
            value.requires_grad = False
        for key, value in self.get_trainable_params().items():
            value.requires_grad = True


    def forward(self, examples, labels, images=None):
        if self.use_vllm:
            # The vLLM engine is primarily for inference and doesn't have a 'forward' like nn.Module for training.
            # If you need to get logits for training/loss calculation with a vLLM-compatible setup,
            # it would typically involve a different workflow, possibly by:
            # 1. Using the vLLM model for inference/generation if parts of a larger system.
            # 2. If trying to fine-tune the model that vLLM *could* serve, you'd do that with a standard
            #    Hugging Face Trainer or custom PyTorch loop on the HF model *before* serving with vLLM.
            raise NotImplementedError("forward() pass is not implemented when use_vllm is True. vLLM is for inference.")

        # Fallback to original logic if self.llma is somehow None when not using vLLM (should not happen)
        if self.llma is None:
            raise RuntimeError("self.llma is None, but use_vllm is False. Model not properly initialized.")

        with torch.no_grad():
            non_zero_ = torch.count_nonzero(labels, dim=0)
            pos = non_zero_.shape[0] - 1
            while pos >= 0:
                if non_zero_[pos] == 0:
                    pos -= 1
                else:
                    break
            examples = examples[:, :pos+1]
            labels = labels[:, :pos+1]


        output = self.llma(examples, images)
        output = output[:, :-1, :]
        labels = labels[:, 1:]


        if labels.sum() == 0:
           c_loss = output.mean() * 0
        else:
            c_loss = self.criterion(output.reshape(-1, self.tokenizer.n_words), labels.flatten())
        return c_loss


    @ torch.inference_mode()
    def generate(
        self,
        prompts: List[str], # Changed from prompt_tokens to List[str] to align with vLLM and simplify
        images: Optional[List] = None, # Made images Optional and ensured List type hint consistency
        max_gen_len: int = 64, # Added default from your vLLM version
        temperature: float = 0.8,
        top_p: float = 0.95,
        return_logits: bool = False # From original generate
    ) -> List[str]: # Original returned List[str]

        if self.llma is None:
             raise RuntimeError("self.llma is None, but use_vllm is False. Model not properly initialized for standard generation.")

        bsz = len(prompts)
        params = self.llma.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(
            x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        max_seq_len = params.max_seq_len
        if images is not None and hasattr(self.llma, 'image_words') and self.llma.image_words > 0:
            max_seq_len -= self.llma.image_words

        total_len = min(max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), getattr(self.tokenizer, 'pad_id', 0)).cuda().long()
        input_text_mask = torch.full((bsz, total_len), False).cuda()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
            input_text_mask[k, : len(t)] = True
        start_pos = min_prompt_size
        prev_pos = 0

        if return_logits:
            return self.llma.forward_inference(tokens[:, :start_pos], prev_pos, images if prev_pos == 0 and images else None)
    
        for cur_pos in range(start_pos, total_len):
            current_images = images if prev_pos == 0 and images else None
            logits = self.llma.forward_inference(tokens[:, prev_pos:cur_pos], prev_pos, current_images)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )

            tokens[:, cur_pos] = next_token
            
            if bsz == 1 and hasattr(self.tokenizer, 'eos_id') and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t_list in enumerate(tokens.tolist()):
            # if i == 0:
            #     no_process_decoded_tokens = self.tokenizer.decode(t_list[len(prompt_tokens[i]): -1])
            #     print(f"no_process_decoded_tokens[0]: {no_process_decoded_tokens}")
            gen_tokens_list = t_list[len(prompt_tokens[i]): len(prompt_tokens[i]) + max_gen_len]
            if hasattr(self.tokenizer, 'eos_id'):
                try:
                    eos_idx = gen_tokens_list.index(self.tokenizer.eos_id)
                    gen_tokens_list = gen_tokens_list[:eos_idx]
                except ValueError:
                    pass
            decoded.append(self.tokenizer.decode(gen_tokens_list))
        return decoded



    @ torch.inference_mode()
    def generate_with_vllm(
        self,
        prompts: List[str],
        num_candidates: int = 5,
        max_gen_len: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = -1,
        use_beam_search: bool = False,
        length_penalty: float = 1.0,
    ) -> Tuple[List[List[str]], List[str]]:
        """
        Generates multiple candidate sequences for each prompt using vLLM.
        Args:
            prompts: A list of text prompts.
            num_candidates: Number of candidate sequences to generate for each prompt.
            max_gen_len: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling p.
            top_k: Top-k sampling k.
            use_beam_search: Whether to use beam search. If True, num_candidates acts as num_beams.
            length_penalty: Length penalty for beam search.
        Returns:
            A list of lists of strings, where the outer list corresponds to prompts
            and the inner list contains the generated candidate texts.
        """
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed or not found. Cannot use `generate_with_vllm`.")
        
        if self.vllm_engine_instance is None:
            raise RuntimeError(
                "vLLM engine not initialized. Please initialize MetaModel with use_vllm=True."
            )
        
        engine = self.vllm_engine_instance
        

        prompt_token_ids = [self.tokenizer.encode(x, add_special_tokens=True) for x in prompts]
 
        
        # Prepare stop_token_ids for SamplingParams
        current_stop_token_ids = []
        eos_id_val = None

        # Try 'eos_token_id' (standard for HF/vLLM tokenizers)
        if hasattr(self.tokenizer, 'eos_token_id'):
            eos_id_val = getattr(self.tokenizer, 'eos_token_id', None)
            if eos_id_val is not None:
                 print(f"Info: Using EOS token ID {eos_id_val} (from .eos_token_id) for vLLM stopping.")

        # If not found via 'eos_token_id', try 'eos_id' (used by custom Tokenizer or as a fallback)
        if eos_id_val is None and hasattr(self.tokenizer, 'eos_id'):
            eos_id_val = getattr(self.tokenizer, 'eos_id', None)
            if eos_id_val is not None:
                print(f"Info: Using EOS token ID {eos_id_val} (from .eos_id) for vLLM stopping.")
        
        if eos_id_val is not None:
            current_stop_token_ids.append(eos_id_val)
        else:
            print("Warning: EOS token ID not found in self.tokenizer. vLLM will rely on max_gen_len or other stop criteria.")
        
        sampling_params_dict = {
            "n": num_candidates,
            "temperature": 0.0 if use_beam_search else temperature,
            "top_p": 1.0 if use_beam_search else top_p,
            "max_tokens": max_gen_len,
            "use_beam_search": use_beam_search,
            # "stop": ["\nInput"]
        }

        # # Add gathered stop token IDs to sampling_params_dict if any were found
        # if current_stop_token_ids:
        #     sampling_params_dict["stop_token_ids"] = current_stop_token_ids

        if top_k != -1 and not use_beam_search : # top_k is not used with beam search
            sampling_params_dict["top_k"] = top_k
        if use_beam_search:
            sampling_params_dict["length_penalty"] = length_penalty
            sampling_params_dict["best_of"] = num_candidates
            sampling_params_dict["n"] = num_candidates

        assert VLLM_SamplingParams is not None, "vLLM SamplingParams class not available"
        sampling_params = VLLM_SamplingParams(**sampling_params_dict)


        # 直接传递 token IDs 给 vLLM
        request_outputs = engine.generate(
            prompt_token_ids=prompt_token_ids,  # 而不是 prompts
            sampling_params=sampling_params
        )

        all_output_candidates_by_batch = []
        all_internal_prompt_by_batch = []
        for output in request_outputs:
            # 从 vLLM 的输出对象中获取它实际使用的 prompt
            internal_prompt = output.prompt 
            output_candidates = []
            for candidate in output.outputs:
                output_candidates.append(candidate.text)
            all_output_candidates_by_batch.append(output_candidates)
            all_internal_prompt_by_batch.append(internal_prompt)
        return all_output_candidates_by_batch, all_internal_prompt_by_batch


    @ torch.inference_mode()
    def stream_generate(
        self,
        prompt: str,
        images: Optional[torch.Tensor],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        if self.use_vllm:
            # vLLM does not have a direct equivalent for streaming generation combined with image inputs
            # in the same way self.llma might. This would require a custom vLLM implementation or adapter.
            raise NotImplementedError("stream_generate is not implemented for use_vllm=True in its current form.")
        
        if self.llma is None:
            raise RuntimeError("self.llma is None, but use_vllm is False. Model not properly initialized for stream_generate.")

        params = self.llma.params

        prompt_tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        
        max_seq_len = params.max_seq_len
        if images is not None and hasattr(self.llma, 'image_words') and self.llma.image_words > 0:
            max_seq_len -= self.llma.image_words

        max_prompt_size = max_seq_len - max_gen_len
        if len(prompt_tokens) > max_prompt_size:
            prompt_tokens = prompt_tokens[-max_prompt_size:]

        prompt_size = len(prompt_tokens)
        total_len = min(max_seq_len, max_gen_len + prompt_size)

        tokens = torch.full([total_len], getattr(self.tokenizer, 'pad_id', 0)).cuda().long()

        tokens[:len(prompt_tokens)] = torch.tensor(prompt_tokens).long()
        start_pos = prompt_size
        prev_pos = 0
        generate_until = start_pos
        
        for cur_pos_stream in range(start_pos, total_len):
            current_images_stream = images if prev_pos == 0 and images else None
            logits = self.llma.forward_inference(tokens[None, prev_pos:cur_pos_stream], prev_pos, current_images_stream)
            
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token_item = self.sample_top_p(probs, top_p).item()
            else:
                next_token_item = torch.argmax(logits, dim=-1).item()

            if hasattr(self.tokenizer, 'eos_id') and next_token_item == self.tokenizer.eos_id:
                break

            tokens[cur_pos_stream] = next_token_item
            prev_pos = cur_pos_stream
            generate_until = cur_pos_stream + 1
            yield {"text": self.tokenizer.decode(tokens[start_pos:generate_until].tolist()), "end_of_content": False}

        yield {"text": self.tokenizer.decode(tokens[start_pos:generate_until].tolist()), "end_of_content": True}

    def sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token


    def get_image_words(self):
        if self.use_vllm:
            print("Warning: get_image_words() called with use_vllm=True. vLLM does not use 'image_words' concept. Returning 0.")
            return 0
        if self.llma and hasattr(self.llma, 'image_words'):
            return self.llma.image_words
        return 0    

    @ torch.inference_mode()
    def generate_with_hf(
        self,
        hf_model_path: str,
        prompts: List[str],
        max_gen_len: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True
    ) -> List[List[str]]:
        """
        使用HuggingFace transformers直接加载HF格式模型进行生成
        这个方法避免了vLLM的影响，可以用来验证是否是模型转换问题还是vLLM生成策略问题
        
        Args:
            hf_model_path: HF格式模型的路径
            prompts: 输入提示列表
            max_gen_len: 最大生成长度
            temperature: 采样温度
            top_p: nucleus采样参数
            top_k: top-k采样参数
            do_sample: 是否使用采样（False为贪婪解码）
            num_return_sequences: 每个prompt返回的序列数量
            pad_token_id: padding token id
            eos_token_id: end of sequence token id
            use_cache: 是否使用KV缓存
            
        Returns:
            List[List[str]]: 每个prompt对应的生成结果列表
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers library is required for generate_with_hf. Please install it with: pip install transformers")
        
        print(f"🔧 Loading HF model from: {hf_model_path}")
        
        # 加载HF模型和tokenizer
        try:
            hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
            hf_model = AutoModelForCausalLM.from_pretrained(
                hf_model_path,
                torch_dtype=torch.float16,  # 使用float16以节省内存
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # 设置pad_token如果没有的话
            if hf_tokenizer.pad_token is None:
                if hf_tokenizer.eos_token is not None:
                    hf_tokenizer.pad_token = hf_tokenizer.eos_token
                    print("📝 Set pad_token to eos_token")
                else:
                    hf_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    print("📝 Added new pad_token: [PAD]")
            
            # 获取特殊token的ID
            if pad_token_id is None:
                pad_token_id = hf_tokenizer.pad_token_id
            if eos_token_id is None:
                eos_token_id = hf_tokenizer.eos_token_id
                
            print(f"📊 Model loaded successfully!")
            print(f"📊 Vocab size: {hf_tokenizer.vocab_size}")
            print(f"📊 Pad token ID: {pad_token_id}")
            print(f"📊 EOS token ID: {eos_token_id}")
            
        except Exception as e:
            print(f"❌ Failed to load HF model: {e}")
            raise e
        
        all_results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n🔄 Processing prompt {i+1}/{len(prompts)}")
            print(f"📝 Prompt: '{prompt}'")
            
            try:
                # Tokenize输入
                inputs = hf_tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                print(f"📊 Input tokens: {inputs['input_ids'].shape[1]}")
                
                # 生成参数
                generation_kwargs = {
                    "max_new_tokens": max_gen_len,
                    "do_sample": do_sample,
                    "temperature": temperature if do_sample else None,
                    "top_p": top_p if do_sample else None,
                    "top_k": top_k if do_sample else None,
                    "num_return_sequences": num_return_sequences,
                    "pad_token_id": pad_token_id,
                    "eos_token_id": eos_token_id,
                    "use_cache": use_cache,
                    "return_dict_in_generate": True,
                    "output_scores": False  # 不需要分数以节省内存
                }
                
                # 移除None值
                generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
                
                print(f"🎯 Generation params: {generation_kwargs}")
                
                # 生成
                with torch.no_grad():
                    outputs = hf_model.generate(**inputs, **generation_kwargs)
                
                # 解码结果
                prompt_results = []
                generated_sequences = outputs.sequences
                
                for seq_idx in range(num_return_sequences):
                    # 获取生成的token（去除输入部分）
                    generated_tokens = generated_sequences[seq_idx][inputs['input_ids'].shape[1]:]
                    
                    # 解码
                    generated_text = hf_tokenizer.decode(
                        generated_tokens, 
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    prompt_results.append(generated_text)
                    
                    print(f"📤 Result {seq_idx+1}: '{generated_text}'")
                    print(f"📊 Generated tokens: {len(generated_tokens)}")
                
                all_results.append(prompt_results)
                
            except Exception as e:
                print(f"❌ Generation failed for prompt {i+1}: {e}")
                # 返回错误信息而不是崩溃
                all_results.append([f"ERROR: {str(e)}"] * num_return_sequences)
        
        print(f"\n✅ HF generation completed for {len(prompts)} prompts")
        return all_results

    @ torch.inference_mode()
    def generate_multi_candidate(
        self,
        prompts: List[str],
        images: Optional[List] = None,
        max_gen_len: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        num_candidates: int = 1,
    ) -> List[List[str]]:
        """
        Generates multiple candidate sequences for each prompt without vLLM or HF transformers.
        This is achieved by expanding the input batch and running a single auto-regressive generation.
        """
        if self.llma is None:
            raise RuntimeError("self.llma is None. Model not properly initialized for generation.")
        
        if num_candidates <= 0:
            raise ValueError("num_candidates must be a positive integer.")

        # 1. Expand inputs for batch processing
        original_bsz = len(prompts)
        expanded_prompts = [p for p in prompts for _ in range(num_candidates)]
        if images is not None:
            expanded_images = [img for img in images for _ in range(num_candidates)]
        else:
            expanded_images = None
        
        bsz = len(expanded_prompts)
        params = self.llma.params
        if bsz > params.max_batch_size:
             print(f"Warning: Batch size {bsz} (prompts * candidates) exceeds max_batch_size {params.max_batch_size}. This may cause memory issues.")

        # 2. Tokenization
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in expanded_prompts]
        max_prompt_size = max([len(t) for t in prompt_tokens]) if prompt_tokens else 0
        min_prompt_size = min([len(t) for t in prompt_tokens]) if prompt_tokens else 0


        # 3. Prepare tensors for generation loop
        max_seq_len = params.max_seq_len
        if expanded_images is not None and hasattr(self.llma, 'image_words') and self.llma.image_words > 0:
            max_seq_len -= self.llma.image_words
        total_len = min(max_seq_len, max_gen_len + max_prompt_size)
        
        tokens = torch.full((bsz, total_len), getattr(self.tokenizer, 'pad_id', 0)).cuda().long()
        input_text_mask = torch.full((bsz, total_len), False, dtype=torch.bool).cuda()

        for k, t in enumerate(prompt_tokens):
            if len(t) > total_len:
                t = t[:total_len]
            tokens[k, : len(t)] = torch.tensor(t).long()
            input_text_mask[k, : len(t)] = True
        
        start_pos = min_prompt_size
        prev_pos = 0

        # 4. Auto-regressive generation loop
        for cur_pos in range(start_pos, total_len):
            current_images = expanded_images if prev_pos == 0 and expanded_images is not None else None
            logits = self.llma.forward_inference(tokens[:, prev_pos:cur_pos], prev_pos, current_images)
            
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else: # Greedy decoding
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        # 5. Decode and group results
        all_decoded = []
        for i, t_list in enumerate(tokens.tolist()):
            prompt_len = len(prompt_tokens[i])
            gen_tokens_list = t_list[prompt_len : prompt_len + max_gen_len]
            
            if hasattr(self.tokenizer, 'eos_id'):
                try:
                    eos_idx = gen_tokens_list.index(self.tokenizer.eos_id)
                    gen_tokens_list = gen_tokens_list[:eos_idx]
                except ValueError:
                    pass
            all_decoded.append(self.tokenizer.decode(gen_tokens_list))
        
        output_by_prompt = [all_decoded[i:i + num_candidates] for i in range(0, len(all_decoded), num_candidates)]
        
        return output_by_prompt

