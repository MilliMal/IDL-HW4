"""
Profile validation performance to identify bottlenecks in greedy sequence generation.
"""
import torch
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from hw4lib.data import ASRDataset
from hw4lib.model import EncoderDecoderTransformer
from hw4lib.decoding.sequence_generator import SequenceGenerator
from torch.utils.data import DataLoader, Subset

def profile_greedy_validation(num_batches=2, batch_size=4):
    """Profile greedy validation to identify bottlenecks"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load a small dataset
    print("\n[1/5] Loading dataset...")
    start = time.time()
    val_dataset = ASRDataset(
        data_dir="hw4_data_subset/hw4p2_data/dev-clean",
        char_file="hw4_data_subset/hw4p2_data/char_set.txt",
        mode="val"
    )
    subset_dataset = Subset(val_dataset, list(range(min(num_batches * batch_size, len(val_dataset)))))
    val_loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True
    )
    print(f"Loaded {len(subset_dataset)} samples in {time.time() - start:.2f}s")
    
    # Initialize model
    print("\n[2/5] Initializing model...")
    start = time.time()
    model = EncoderDecoderTransformer(
        d_model=256,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=1024,
        vocab_size=val_dataset.vocab_size,
        max_len=5000,
        dropout=0.1,
        device=device,
        activation='gelu'
    )
    model = model.to(device)
    model.eval()
    print(f"Model initialized in {time.time() - start:.2f}s")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Profile encoding
    print("\n[3/5] Profiling encoding step...")
    total_encode_time = 0
    num_samples = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(val_loader):
            start = time.time()
            feats, _, targets_golden, feat_lengths, _ = batch
            feats = feats.to(device)
            feat_lengths = feat_lengths.to(device)
            
            encoder_output, pad_mask_src, _, _ = model.encode(feats, feat_lengths)
            torch.cuda.synchronize() if device == "cuda" else None
            encode_time = time.time() - start
            total_encode_time += encode_time
            num_samples += feats.size(0)
            print(f"  Batch {batch_idx}: {encode_time:.3f}s for {feats.size(0)} samples")
    
    print(f"Total encoding time: {total_encode_time:.2f}s ({total_encode_time/num_samples:.3f}s per sample)")
    
    # Profile decoding (scoring function)
    print("\n[4/5] Profiling decoding/scoring step...")
    text_max_len = val_dataset.text_max_len
    print(f"Max text length: {text_max_len}")
    
    total_decode_time = 0
    total_score_calls = 0
    
    with torch.inference_mode():
        for batch_idx, batch in enumerate(val_loader):
            feats, _, targets_golden, feat_lengths, _ = batch
            feats = feats.to(device)
            feat_lengths = feat_lengths.to(device)
            
            batch_size = feats.size(0)
            encoder_output, pad_mask_src, _, _ = model.encode(feats, feat_lengths)
            
            # Simulate the score function that would be called during generation
            def get_score(x):
                start_score = time.time()
                asr_logits = model.decode(x, encoder_output, pad_mask_src, src_pad_mask=pad_mask_src)
                torch.cuda.synchronize() if device == "cuda" else None
                end_score = time.time()
                return asr_logits
            
            # Create initial prompts
            prompts = torch.full(
                (batch_size, 1),
                val_dataset.tokenizer.sos_id,
                dtype=torch.long,
                device=device
            )
            
            # Simulate greedy generation
            x = prompts
            generation_start = time.time()
            
            for gen_step in range(min(50, text_max_len - 1)):  # Profile only first 50 steps
                start_decode = time.time()
                next_scores = get_score(x)
                torch.cuda.synchronize() if device == "cuda" else None
                decode_time = time.time() - start_decode
                total_decode_time += decode_time
                total_score_calls += 1
                
                next_tokens = torch.argmax(next_scores, dim=-1)
                x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)
                
                if gen_step % 10 == 0:
                    print(f"  Step {gen_step}: decode time {decode_time:.4f}s, seq len: {x.size(1)}")
            
            generation_time = time.time() - generation_start
            print(f"Batch {batch_idx}: Full generation took {generation_time:.3f}s ({generation_time/50:.3f}s per token)")
    
    print(f"\nTotal decode time: {total_decode_time:.2f}s")
    print(f"Total score function calls: {total_score_calls}")
    print(f"Average time per score call: {total_decode_time/total_score_calls:.4f}s")
    
    # Profile post-processing
    print("\n[5/5] Profiling post-processing...")
    start = time.time()
    generator = SequenceGenerator(
        score_fn=None,
        tokenizer=val_dataset.tokenizer,
        max_length=text_max_len,
        device=device
    )
    
    # Generate a dummy sequence
    test_seq = torch.randint(0, val_dataset.vocab_size, (4, 50), device=device)
    test_seq[:, 0] = val_dataset.tokenizer.sos_id
    
    for i in range(100):
        post_processed = generator.post_process_sequence(test_seq, val_dataset.tokenizer)
    
    post_process_time = time.time() - start
    print(f"Post-processing 100 sequences: {post_process_time:.3f}s")
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Encoding: {total_encode_time:.2f}s (efficient)")
    print(f"Decoding (scoring): {total_decode_time:.2f}s (potential bottleneck)")
    print(f"Estimated full validation time:")
    estimated_steps = text_max_len - 1
    estimated_total = (total_decode_time / total_score_calls) * estimated_steps * len(subset_dataset) / batch_size
    print(f"  ~{estimated_total/60:.1f} minutes for {len(subset_dataset)} samples at {text_max_len} tokens")
    
if __name__ == "__main__":
    profile_greedy_validation(num_batches=2, batch_size=4)
