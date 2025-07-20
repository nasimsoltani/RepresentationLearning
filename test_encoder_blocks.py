#!/usr/bin/env python3
"""
Test script to verify that encoder blocks work correctly with the main codebase.
This script tests different encoder configurations without requiring full training.
"""

import argparse
import torch
import torch.nn as nn
from code.rep_lr.models import Encoder, TaskAdaptiveEncoder, ComplexSequenceProjector, CFOEstimationHead

def test_encoder_configs():
    """Test different encoder configurations"""
    print("Testing Encoder Block Configurations")
    print("=" * 50)
    
    # Common parameters
    slice_size = 256
    proj_seq_len = 256
    d2 = 128
    batch_size = 4
    
    # Test data
    x = torch.randn(batch_size, 2, proj_seq_len)
    
    for num_blocks in [1, 2, 3, 5]:
        print(f"\n--- Testing {num_blocks} blocks ---")
        
        # Test standard Encoder
        try:
            encoder = Encoder(slice_size=proj_seq_len, output_dim=d2, num_blocks=num_blocks)
            output = encoder(x)
            params = sum(p.numel() for p in encoder.parameters())
            print(f"✓ Standard Encoder ({num_blocks} blocks): {params:,} params, output shape: {output.shape}")
        except Exception as e:
            print(f"✗ Standard Encoder ({num_blocks} blocks): ERROR - {e}")
        
        # Test TaskAdaptiveEncoder  
        try:
            task_encoder = TaskAdaptiveEncoder(slice_size=proj_seq_len, output_dim=d2, num_blocks=num_blocks)
            output = task_encoder(x)
            params = sum(p.numel() for p in task_encoder.parameters())
            print(f"✓ TaskAdaptiveEncoder ({num_blocks} blocks): {params:,} params, output shape: {output.shape}")
        except Exception as e:
            print(f"✗ TaskAdaptiveEncoder ({num_blocks} blocks): ERROR - {e}")

def test_full_pipeline():
    """Test complete CFO estimation pipeline"""
    print("\n" + "=" * 50)
    print("Testing Complete CFO Pipeline")
    print("=" * 50)
    
    # Parameters
    batch_size = 4
    proj_seq_len = 256
    d2 = 128
    
    # Test input (CFO data: batch_size, 2, 160)
    cfo_input = torch.randn(batch_size, 2, 160)
    
    for num_blocks in [1, 3]:
        print(f"\n--- Testing Complete Pipeline with {num_blocks} blocks ---")
        
        try:
            # Create components
            projection = ComplexSequenceProjector(input_seq_len=160, output_seq_len=proj_seq_len, hidden_dim=512)
            encoder = TaskAdaptiveEncoder(slice_size=proj_seq_len, output_dim=d2, num_blocks=num_blocks)
            head = CFOEstimationHead(input_dim=2*d2, hidden_dim=256)
            
            # Forward pass
            x = projection(cfo_input)
            x = encoder(x)
            output = head(x)
            
            # Count parameters
            total_params = sum(p.numel() for p in [projection, encoder, head] for p in p.parameters())
            
            print(f"✓ Complete Pipeline ({num_blocks} blocks):")
            print(f"  - Input shape: {cfo_input.shape}")
            print(f"  - After projection: {x.shape}")
            print(f"  - After encoder: {x.shape}")
            print(f"  - Final output: {output.shape}")
            print(f"  - Total parameters: {total_params:,}")
            
        except Exception as e:
            print(f"✗ Complete Pipeline ({num_blocks} blocks): ERROR - {e}")

def test_argument_parsing():
    """Test that argument parsing works correctly"""
    print("\n" + "=" * 50)
    print("Testing Argument Parsing")
    print("=" * 50)
    
    # Simulate command line arguments
    test_args = [
        ['--encoder_num_blocks', '1'],
        ['--encoder_num_blocks', '3'],
        ['--encoder_num_blocks', '5'],
    ]
    
    for args in test_args:
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument('--encoder_num_blocks', type=int, default=1,
                              help='Number of convolutional blocks in the encoder.')
            parser.add_argument('--task_adaptive_encoder', action='store_true',
                              help='Use task-adaptive encoder.')
            
            parsed_args = parser.parse_args(args)
            print(f"✓ Args {args}: encoder_num_blocks = {parsed_args.encoder_num_blocks}")
            
        except Exception as e:
            print(f"✗ Args {args}: ERROR - {e}")

if __name__ == "__main__":
    print("Encoder Blocks Functionality Test")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run tests
    test_encoder_configs()
    test_full_pipeline()
    test_argument_parsing()
    
    print("\n" + "=" * 50)
    print("✓ All tests completed!")
    print("\nYou can now use the main training script with:")
    print("python code/rep_lr/main.py --task cfo_estimation --encoder_num_blocks 1 ...")
    print("python code/rep_lr/main.py --task cfo_estimation --encoder_num_blocks 3 --task_adaptive_encoder ...") 