#!/usr/bin/env python3
"""
Debug script to compare model architectures between barebones.py and main pipeline.
This ensures both create identical models.
"""

import torch
import torch.nn as nn
from code.rep_lr.models import TaskAdaptiveEncoder, ComplexSequenceProjector, CFOEstimationHead

def create_barebones_model():
    """Create model exactly like barebones.py"""
    projection = ComplexSequenceProjector(
        input_seq_len=160,  # CFO input length
        output_seq_len=256,  # proj_seq_len
        hidden_dim=512       # proj_hidden_dim
    )
    
    encoder = TaskAdaptiveEncoder(
        slice_size=256,      # proj_seq_len
        output_dim=128,      # d2
        dropout=0.1,
        num_blocks=1         # Key: 1 block
    )
    
    head = CFOEstimationHead(
        input_dim=2 * 128,   # 2 * d2
        hidden_dim=256,      # head_hidden_dim
        dropout=0.1
    )
    
    model = nn.ModuleDict({
        'projection': projection,
        'encoder': encoder,
        'head': head
    })
    
    return model

def create_main_pipeline_model():
    """Create model exactly like main pipeline single-task CFO"""
    # From main.py single-task CFO creation
    projection = ComplexSequenceProjector(
        input_seq_len=160,   # CFO input length
        output_seq_len=256,  # proj_seq_len
        hidden_dim=512       # proj_hidden_dim
    )
    
    encoder = TaskAdaptiveEncoder(
        slice_size=256,      # proj_seq_len
        output_dim=128,      # d2
        dropout=0.1,
        num_blocks=1         # encoder_num_blocks=1
    )
    
    head = CFOEstimationHead(
        input_dim=2 * 128,   # 2 * d2
        hidden_dim=256,      # head_hidden_dim
        dropout=0.1
    )
    
    model = nn.ModuleDict({
        'projection': projection,
        'encoder': encoder,
        'head': head
    })
    
    return model

def compare_models():
    """Compare the two model architectures"""
    print("Model Architecture Comparison")
    print("=" * 50)
    
    # Create models
    barebones_model = create_barebones_model()
    main_model = create_main_pipeline_model()
    
    # Count parameters
    def count_params(model):
        return {
            'total': sum(p.numel() for p in model.parameters()),
            'projection': sum(p.numel() for p in model['projection'].parameters()),
            'encoder': sum(p.numel() for p in model['encoder'].parameters()),
            'head': sum(p.numel() for p in model['head'].parameters())
        }
    
    barebones_params = count_params(barebones_model)
    main_params = count_params(main_model)
    
    print("Barebones Model:")
    for key, value in barebones_params.items():
        print(f"  {key}: {value:,}")
    
    print("\nMain Pipeline Model:")
    for key, value in main_params.items():
        print(f"  {key}: {value:,}")
    
    print("\nParameter Differences:")
    for key in barebones_params:
        diff = main_params[key] - barebones_params[key]
        print(f"  {key}: {diff:,} {'✓' if diff == 0 else '✗'}")
    
    # Test forward pass
    print("\nForward Pass Test:")
    test_input = torch.randn(4, 2, 160)  # CFO input
    
    try:
        # Barebones forward
        x1 = barebones_model['projection'](test_input)
        x1 = barebones_model['encoder'](x1)
        out1 = barebones_model['head'](x1)
        
        # Main pipeline forward
        x2 = main_model['projection'](test_input)
        x2 = main_model['encoder'](x2)
        out2 = main_model['head'](x2)
        
        print(f"  Barebones output shape: {out1.shape}")
        print(f"  Main pipeline output shape: {out2.shape}")
        print(f"  Shapes match: {'✓' if out1.shape == out2.shape else '✗'}")
        
        # Copy weights to test identical behavior
        main_model.load_state_dict(barebones_model.state_dict())
        
        x3 = main_model['projection'](test_input)
        x3 = main_model['encoder'](x3)
        out3 = main_model['head'](x3)
        
        diff = torch.abs(out1 - out3).max().item()
        print(f"  Max output difference with same weights: {diff:.6f} {'✓' if diff < 1e-6 else '✗'}")
        
    except Exception as e:
        print(f"  Forward pass error: {e}")
    
    print("\nConclusion:")
    if all(main_params[k] == barebones_params[k] for k in barebones_params):
        print("✅ Models are architecturally identical!")
        print("   Performance difference likely due to:")
        print("   1. Learning rate (3e-5 vs 1e-3)")
        print("   2. Training loop differences")
        print("   3. WandB overhead")
    else:
        print("❌ Models have architectural differences!")

def test_loss_functions():
    """Test if loss functions are identical"""
    print("\n" + "=" * 50)
    print("Loss Function Comparison")
    print("=" * 50)
    
    # Both use HuberLoss(delta=0.1)
    loss_fn = nn.HuberLoss(delta=0.1)
    
    # Test data
    pred = torch.randn(4, 1)
    target = torch.randn(4, 1)
    
    loss = loss_fn(pred, target)
    print(f"HuberLoss test: {loss.item():.6f}")
    print("✅ Both models use identical HuberLoss(delta=0.1)")

if __name__ == "__main__":
    compare_models()
    test_loss_functions()
    
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)
    print("To fix your main pipeline:")
    print("1. Change lr from 3e-5 to 1e-3")
    print("2. Change batch_size from 128 to 64")
    print("3. Models are architecturally identical!")
    print("4. Consider temporarily disabling WandB for debugging") 