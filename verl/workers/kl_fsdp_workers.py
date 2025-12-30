
import torch
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl import DataProto
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils.profiler import simple_timer, reduce_timing, topk_reduce_ratio_min_max, log_gpu_memory_usage
from verl.utils.ray_utils import get_event_loop
from verl.utils.device import get_device_id, get_torch_device
import logging

logger = logging.getLogger(__file__)

class KLActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    Subclass of ActorRolloutRefWorker that extracts attention maps during generation for KL Anchoring.
    """
    
    def __init__(self, config, role, **kwargs):
        super().__init__(config, role, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def generate_sequences_with_density(self, prompts: DataProto):
        # ... Essentially copy of RSActorRolloutRefWorker ...
        assert self._is_rollout
        prompts = prompts.to(get_device_id())

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id if self.generation_config else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate = {}
        if self._is_actor: 
            loop = get_event_loop()
            loop.run_until_complete(self.rollout_mode())
        
        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)
            
        # Attention Extraction
        if self._is_actor:
             loop.run_until_complete(self.trainer_mode())
             
             with simple_timer("attention_extraction", timing_generate):
                with torch.no_grad():
                    # Concatenate prompt + response
                    batch_prompts = output.batch['prompts']
                    batch_responses = output.batch['responses']
                    
                    full_input_ids = torch.cat([batch_prompts, batch_responses], dim=1)
                    pad_token_id = self.tokenizer.pad_token_id
                    full_attention_mask = (full_input_ids != pad_token_id).long()
                    
                    full_input_ids = full_input_ids.to(get_device_id())
                    full_attention_mask = full_attention_mask.to(get_device_id())
                    
                    model = self.actor_module_fsdp
                    res = model(
                        input_ids=full_input_ids,
                        attention_mask=full_attention_mask,
                        output_attentions=True,
                        use_cache=False 
                    )
                    
                    last_layer_attn = res.attentions[-1]
                    avg_attn = last_layer_attn.mean(dim=1)
                    density_scores = avg_attn.sum(dim=1) # Sum over columns (dim 1?)
                    # Wait, attention map is (batch, heads, q_len, k_len).
                    # avg_attn is (batch, q_len, k_len).
                    # "sum across columns, how much it contributes to the final answer"
                    # If we look at A[i, j] = P(i attends to j).
                    # Column sum means \sum_i A[i, j]. 
                    # Yes, sum(dim=1) of (batch, q_len, k_len)?
                    # If dim 1 is Query (Row), dim 2 is Key (Column).
                    # sum(dim=1) sums over Rows (Queries). Result is (batch, k_len).
                    # Yes, density_scores = avg_attn.sum(dim=1).
                    
                    p_len = batch_prompts.shape[1]
                    response_density = density_scores[:, p_len:]
                    
                    output.batch['density_scores'] = response_density

        if self._is_actor:
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
