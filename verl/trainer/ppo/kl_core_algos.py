
import torch
import numpy as np
from collections import defaultdict
from typing import Optional, Tuple, List
from verl.trainer.config import AlgoConfig
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss

def select_core_windows(density_scores: torch.Tensor, window_size: int, min_windows: int, max_windows: int) -> torch.Tensor:
    """
    Selects top disjoint windows based on density scores.
    Reused from Reward Shaping implementation but standalone here.
    """
    seq_len = density_scores.size(0)
    if seq_len < window_size:
        return torch.zeros_like(density_scores, dtype=torch.bool)
    
    cumsum = torch.cumsum(torch.cat([torch.zeros(1, device=density_scores.device), density_scores]), dim=0)
    window_scores = cumsum[window_size:] - cumsum[:-window_size] 
    
    num_possible_windows = window_scores.size(0)
    scores = window_scores.clone()
    mask = torch.ones_like(scores, dtype=torch.bool)
    final_indices = []
    
    # We use a threshold of 0 (mean) for Normalized scores to determine "lower boundary"
    for _ in range(max_windows):
        valid_scores = scores.masked_fill(~mask, -float('inf'))
        if valid_scores.max() == -float('inf'):
            break
            
        max_val, max_idx = torch.max(valid_scores, dim=0)
        max_idx = max_idx.item()
        
        # Stop if score is below average (0), unless we haven't met min_windows
        if len(final_indices) >= min_windows and max_val < 0:
            break
            
        final_indices.append(max_idx)
        
        start_mask = max(0, max_idx - window_size + 1)
        end_mask = min(num_possible_windows, max_idx + window_size)
        mask[start_mask:end_mask] = False
        
    core_mask = torch.zeros(seq_len, dtype=torch.bool, device=density_scores.device)
    for idx in final_indices:
        core_mask[idx : idx + window_size] = True
    return core_mask

def kl_penalty_forward_approx(logprob, ref_logprob):
    # Standard KL approx: logP - logRef
    return logprob - ref_logprob

def apply_selective_kl_anchoring(
    batch,
    kl_ctrl,
    window_size: int = 10,
    min_windows: int = 3,
    max_windows: int = 10,
    sparsity_coeff: float = 0.0, # gamma
    initial_scale: float = 2.0,
    final_scale: float = 1.25,
    global_step: int = 0,
    total_steps: int = 1
):
    """
    Applies Selective KL Anchoring to `token_level_rewards`.
    
    Logic:
    1. Extract Density.
    2. Determine Core/Redundant tokens.
    3. Determine Curriculum Limit.
    4. Apply:
       - Core: R -= Beta * KL
       - Redundant (if Len > Limit): R -= -Gamma * LogP (i.e. R += Gamma * LogP? No, "suppresses logprob". 
          User said: "Gamma * log pi" in Loss (Minimizing). 
          Loss = ... - Gamma * log pi. 
          Reward = - Loss.
          So Reward = ... + Gamma * log pi.
          If we Maximize Reward, we Maximize Log Pi. Maximize Prob.
          BUT User says "Sparsity Penalty... to unlearn". 
          "Sparsity" usually minimizes prob.
          If we want to minimize Prob, we want Log Pi to be very negative.
          If Log Pi is very negative (e.g. -100), and we simply add it to reward?
          Reward = -100. Bad reward.
          So if we want to minimize prob, we should punish high prob.
          If we just subtract Gamma * Log Pi?
          R = - Gamma * (-0.1) = +0.1Gamma. (High prob)
          R = - Gamma * (-100) = +100Gamma. (Low prob)
          So Subtracting Gamma * Log Pi gives higher reward to lower probability.
          This matches "Unlearning". 
          So: **Reward Term = - Gamma * log_prob**.
       - Redundant (if Len <= Limit): R -= Beta * KL.
       
    Args:
        batch: DataProto containing 'token_level_scores', 'density_scores', 'old_log_probs', 'ref_log_probs', 'response_mask'
    """
    
    token_level_scores = batch.batch["token_level_scores"]
    density_scores = batch.batch["density_scores"]
    log_prob = batch.batch["old_log_probs"] # Pi_theta (old policy in current PPO step, or current sample policy)
    ref_log_prob = batch.batch["ref_log_probs"] # Pi_ref
    response_mask = batch.batch["response_mask"]
    
    # KL Coef (Beta) from Controller
    beta_base = kl_ctrl.value
    
    # Curriculum
    if total_steps > 0:
        progress = min(1.0, max(0.0, global_step / total_steps))
    else:
        progress = 1.0
    current_scale = initial_scale + (final_scale - initial_scale) * progress
    
    bs, seq_len = density_scores.shape
    new_rewards = token_level_scores.clone()
    
    # Calculate KL term (Standard)
    # kl = log - ref
    kl_div = log_prob - ref_log_prob
    
    for i in range(bs):
        valid_len = int(response_mask[i].sum().item())
        if valid_len == 0:
            continue
            
        # Get scores
        seq_scores = density_scores[i, :valid_len]
        
        # Norm
        seq_scores_log = torch.log(seq_scores + 1e-9)
        mean = seq_scores_log.mean()
        std = seq_scores_log.std() + 1e-6
        norm_scores = (seq_scores_log - mean) / std
        
        # Core Mask
        core_mask = select_core_windows(norm_scores, window_size, min_windows, max_windows) # (valid_len)
        core_len = core_mask.float().sum().item()
        length_limit = core_len * current_scale
        
        T = valid_len
        
        # Determine Beta and Gamma per token
        # Default: Beta = beta_base, Gamma = 0
        
        # Case 2 check: (M_t = 0) AND (CurrentLen > Limit)
        # Note: "CurrentLen" usually means "Total Sequence Length"? Or "Current Token Index"?
        # "If the sequence is too long ... detach KL ... for Redundant Tokens".
        # This implies checking the TOTAL length of the generated sequence.
        # "CurrentLen" in "(CurrentLen - Limit)/CurrentLen" suggests T (total length).
        
        is_too_long = (T > length_limit)
        
        gamma_t = 0.0
        if is_too_long:
            gamma_t = sparsity_coeff * (T - length_limit) / T
        
        # Expand masks to tensor for this seq
        # core_mask is boolean
        
        # Case 1: Core (M_t=1) -> Beta=Base, Gamma=0
        # Case 2: Redundant (M_t=0) & TooLong -> Beta=0, Gamma=Val
        # Case 3: Redundant (M_t=0) & OK -> Beta=Base, Gamma=0
        
        # We can construct Beta Mask and Gamma Mask
        
        # Default Beta Mask is 1.0
        beta_mask = torch.ones_like(core_mask, dtype=torch.float)
        gamma_mask = torch.zeros_like(core_mask, dtype=torch.float)
        
        if is_too_long:
            # For Redundant tokens, Beta becomes 0, Gamma becomes gamma_t
            redundant = ~core_mask
            beta_mask[redundant] = 0.0
            gamma_mask[redundant] = gamma_t
            
        # Apply Logic:
        # Reward -= Beta_base * Beta_Mask * KL
        # Reward -= Gamma_Mask * LogProb
        
        # Note: log_prob[i, :valid_len]
        
        kl_term = beta_base * beta_mask * kl_div[i, :valid_len]
        sparsity_term = gamma_mask * log_prob[i, :valid_len] # Remember we subtract Gamma * logP to penalize high prob
        
        # Update
        # Original Reward (Outcome) usually only at last token?
        # `token_level_scores` usually has outcome at last index and 0 elsewhere?
        # Or dense reward?
        # GRPO usually takes outcome scalars.
        # We modify the token rewards in place.
        
        # r = r - kl - sparsity
        current_r = new_rewards[i, :valid_len]
        new_rewards[i, :valid_len] = current_r - kl_term - sparsity_term
        
    batch.batch["token_level_rewards"] = new_rewards
    
    # Return metrics
    metrics = {
        "kl/mean_beta": beta_base, # Simplified
        # "kl/mean_gamma": gamma_mask.mean().item() # Hard to log without accumulation
    }
    return batch, metrics
