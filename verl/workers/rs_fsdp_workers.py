
import torch
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl import DataProto
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils.profiler import DistProfiler, simple_timer, reduce_timing, topk_reduce_ratio_min_max
from verl.utils.ray_utils import get_event_loop
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.profiler import log_gpu_memory_usage
import logging

logger = logging.getLogger(__file__)

class RSActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    Subclass of ActorRolloutRefWorker that extracts attention maps during generation.
    """
    
    def __init__(self, config, role, **kwargs):
        super().__init__(config, role, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL) # Using default dispatch for simple extension
    def generate_sequences_with_density(self, prompts: DataProto):
        """
        Generates sequences and then runs a forward pass to extract attention density.
        """
        # 1. Standard Generation
        # We can call the parent's generate_sequences, but we need to inject the logic *after* generation 
        # but *before* returning to memory handling or cpu. 
        # The parent method releases GIL or switches modes. 
        # Easier to copy-paste the method and modify. 
        
        assert self._is_rollout
        prompts = prompts.to(get_device_id())

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate = {}
        if self._is_actor: 
            loop = get_event_loop()
            loop.run_until_complete(self.rollout_mode())
        
        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)
            
        # --- NEW LOGIC START ---
        # 2. Attention Extraction
        # We need to run a forward pass on the [prompt + response] to get attentions.
        # "Signal Extraction: Take the last layer's attention map ... average the final layer head, and then sum across columns"
        
        if self._is_actor:
             # If we are in rollout mode, we might need to access the model weights. 
             # self.rollout.model is usually the inference engine (vLLM). 
             # vLLM doesn't easily return attention maps.
             # HOWEVER, we have the FSDP parameters loaded in `self.actor_module_fsdp` (or similar) when in Trainer mode.
             # OR we can assume we verify this on the PPO training model?
             
             # Problem: `generate_sequences` runs in `rollout_mode`.
             # The model weights are in `self.rollout`.
             # vLLM does not support `output_attentions=True` easily for standard `generate`.
             # We might need to switch back to `trainer_mode` (PyTorch model) to run the forward pass?
             # Yes, switching to trainer_mode loads PyTorch weights.
             
             loop.run_until_complete(self.trainer_mode())
             
             # Now run forward pass using PyTorch model
             # Construct full input_ids and attention_mask
             # output.batch contains "prompts", "responses", "input_ids" (maybe), "attention_mask"
             
             # We need to concatenate prompt and response if not already done.
             # `output` from vLLM usually has `prompts` and `responses`.
             # We need to combine them.
             
             with simple_timer("attention_extraction", timing_generate):
                with torch.no_grad():
                    # Concatenate prompt + response
                    # Note: We need to handle padding/masking correctly.
                    # This logic depends on DataProto structure.
                    
                    batch_prompts = output.batch['prompts']
                    batch_responses = output.batch['responses']
                    
                    # Assuming we pad to left for prompts and right for responses or similar? 
                    # Usually we just concat.
                    # But shapes might be ragged if not padded. 
                    # DataProto usually handles tensor dicts.
                    
                    # Let's perform forward pass on `input_ids` + `responses`
                    # We might need to pad if lengths differ.
                    
                    input_ids = []
                    attention_masks = []
                    
                    # We iterate to construct the full sequences for forward pass
                    bs = batch_prompts.shape[0]
                    
                    # Helper to concat
                    full_ids_list = []
                    max_len = 0
                    
                    for i in range(bs):
                        p = batch_prompts[i]
                        r = batch_responses[i]
                        # Remove padding from p if needed (pad_token_id)
                        # Depending on how prompts are stored. 
                        # Assuming prompts are left-padded?
                        # Let's trust simple concatenation for now, assuming standard verified inputs.
                        
                        # Use `torch.cat`
                        # Filter out padding in prompt if necessary
                        # But batch_prompts is already a tensor, likely padded.
                        # We should create a helper or just use the batch directly if aligned.
                        
                        # Simplified: DataProto batch usually holds padded tensors.
                        # `prompts`: (bs, p_len)
                        # `responses`: (bs, r_len)
                        
                        full = torch.cat([p, r], dim=-1)
                        full_ids_list.append(full)
                        max_len = max(max_len, full.shape[-1])
                    
                    # Stack
                    # We assume equal length or existing padding is fine? 
                    # Actually responses are padded to max_response_len. 
                    # Prompts are padded to max_prompt_len.
                    # So simple cat is fine.
                    
                    # NOTE: We need to mask out padding tokens from attention!
                    # `output.batch` should have `attention_mask`... wait, `generate_sequences` returns 
                    # new data which might NOT have full `attention_mask` for the combination yet.
                    
                    full_input_ids = torch.cat([batch_prompts, batch_responses], dim=1)
                    
                    # Create attention mask
                    # 1 for valid, 0 for pad.
                    pad_token_id = self.tokenizer.pad_token_id
                    full_attention_mask = (full_input_ids != pad_token_id).long()
                    
                    # Move to device
                    full_input_ids = full_input_ids.to(get_device_id())
                    full_attention_mask = full_attention_mask.to(get_device_id())
                    
                    # Forward PASS
                    # Use the actor module.
                    # We need `output_attentions=True`
                    
                    model = self.actor_module_fsdp
                    
                    # We only need the forward pass.
                    res = model(
                        input_ids=full_input_ids,
                        attention_mask=full_attention_mask,
                        output_attentions=True,
                        use_cache=False 
                    )
                    
                    # Extract last layer attention
                    # attentions is a tuple of (num_layers) tensors of shape (bs, num_heads, seq_len, seq_len)
                    last_layer_attn = res.attentions[-1] # (bs, num_heads, seq_len, seq_len)
                    
                    # "average the final layer head"
                    # Average over heads
                    avg_attn = last_layer_attn.mean(dim=1) # (bs, seq_len, seq_len)
                    
                    # "sum across columns, how much it contributes to the final answer"
                    # "columns" usually means "how much this token is attended to by others".
                    # A[i, j] is attention from token i to token j.
                    # We want to know importance of token j.
                    # So we sum over i (rows).
                    # "sum across columns" -> sum(dim=0)? 
                    # If A[i, j] = P(j | i), then sum_i A[i, j] is how much j is attended by all i.
                    
                    density_scores = avg_attn.sum(dim=1) # (bs, seq_len)
                    
                    # Store only the response part of density scores?
                    # The reward shaping logic needs to know which score corresponds to which token.
                    # We can store the full density scores and mask later.
                    # BUT `output` data usually only keeps `responses` for some pipelines. 
                    # We should probably align it with `responses`.
                    # The prompt part of density score is usually high. We care about response.
                    
                    p_len = batch_prompts.shape[1]
                    response_density = density_scores[:, p_len:] # (bs, r_len)
                    
                    output.batch['density_scores'] = response_density
             
             # If we switched to trainer mode, we stay there? 
             # The original code switches back to trainer mode at the end anyway.
             pass 
             
        else:
             # If strictly not actor (e.g. standalone rollout), we don't have PyTorch model loaded typically?
             # But `ActorRolloutRefWorker` usually has it.
             # If not, we might fail. Assuming we use standard setup where ActorRolloutRefWorker has the model.
             pass

        # --- NEW LOGIC END ---

        if self._is_actor: # Logic from original
            # loop.run_until_complete(self.trainer_mode()) # Already done above
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)

        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        get_torch_device().empty_cache()
        return output
