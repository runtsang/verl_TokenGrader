
import torch
import unittest
import numpy as np
from verl.trainer.ppo.rs_core_algos import select_core_windows, compute_dense_rewards

class TestRSCoreAlgos(unittest.TestCase):
    def test_select_core_windows(self):
        # Create a simple density score vector
        # Sequence length 20, window size 5
        scores = torch.tensor([
            1.0, 1.0, 1.0, 1.0, 1.0, # Window 1 (Sum 5)
            0.1, 0.1, 0.1, 0.1, 0.1, # Low
            2.0, 2.0, 2.0, 2.0, 2.0, # Window 3 (Sum 10) - Should be picked first
            0.5, 0.5, 0.5, 0.5, 0.5  # Low
        ])
        
        # Norm in test or assume inputs are somewhat normalized or raw
        # The function expects 'density_scores'
        
        # Case 1: Select 2 windows. Should pick W3 and W1.
        mask = select_core_windows(scores, window_size=5, min_windows=1, max_windows=2)
        
        # Expected mask: Indices 0-4 and 10-14 should be True.
        self.assertTrue(torch.all(mask[0:5]))
        self.assertTrue(torch.all(mask[10:14]))
        self.assertFalse(torch.any(mask[5:10]))
        self.assertFalse(torch.any(mask[15:]))
        
    def test_select_core_windows_overlap(self):
        # Test that it doesn't pick overlapping windows
        scores = torch.tensor([
            1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0 # 
        ])
        # W=3
        # Windows: 
        # [0:3] -> 12
        # [1:4] -> 12
        # [2:5] -> 12
        # ...
        
        # If we pick index 2 (10), it covers 2,3,4.
        
        mask = select_core_windows(scores, window_size=3, min_windows=1, max_windows=2)
        # Should pick one window centered around the 10.
        
        self.assertEqual(mask.sum().item(), 3) # Exactly one window size
        
    def test_compute_dense_rewards(self):
        bs = 1
        seq_len = 20
        scores = torch.randn(bs, seq_len)
        mask = torch.ones(bs, seq_len)
        
        # Just ensure it runs and returns tensor
        rewards = compute_dense_rewards(scores, mask)
        self.assertEqual(rewards.shape, (bs, seq_len))

if __name__ == '__main__':
    unittest.main()
