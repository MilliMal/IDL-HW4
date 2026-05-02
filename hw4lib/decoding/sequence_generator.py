import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''
TODO: Implement the `generate_greedy` and optionally the `generate_beam` methods of the `SequenceGenerator` class.

This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
   - Simple but can lead to repetitive or suboptimal outputs
   - Useful for deterministic generation

2. Beam Search: Maintains top-k most likely sequences at each step
   - Explores multiple possible sequences in parallel
   - Often produces higher quality outputs than greedy search
   - More computationally intensive

3. Sampling with Filtering: Uses probabilistic sampling with constraints
   - Temperature: Controls randomness of sampling
   - Top-k: Limits sampling to k most likely tokens
   - Top-p (nucleus): Samples from minimal set of tokens comprising p probability mass
   - Useful for creative and diverse generation

Implementation Notes:
1. Helper Methods:
   - _apply_repeat_penalty: Penalizes repeated tokens
   - _filter_logits: Applies temperature and filtering
   - post_process_sequence: Handles EOS token truncation

2. Generation Methods:
   - generate_greedy: Implements basic greedy decoding
   - generate_beam: Implements beam search
   - generate_sample: Implements filtered sampling

3. Each generation method should:
   - Handle proper input validation
   - Track sequence scores
   - Handle EOS token detection
   - Support early stopping
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sequence generator.
        
        Args:
            score_fn: Function that returns logits for next token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value
        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits
        
        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0/penalty)
                    )
        
        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering to logits."""
        logits = logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits

    def generate_greedy(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")

        scores = torch.zeros(x.size(0), device=x.device)
        finished_flags = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            # Early exit — if all sequences have hit EOS, stop immediately
            if finished_flags.all():
                break

            next_logits = self.score_fn(x)  # (batch_size, vocab_size)
            filtered_logits = self._filter_logits(next_logits, temperature, top_k=0, top_p=1.0)
            filtered_logits = self._apply_repeat_penalty(filtered_logits, x, repeat_penalty)

            # FIX: use log-prob of the chosen token, not raw logit value
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            next_tokens = torch.argmax(log_probs, dim=-1)  # (batch_size,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)  # (batch_size,)

            # For finished sequences, force EOS so we can detect termination,
            # but do NOT keep appending after the loop exits.
            next_tokens = torch.where(
                finished_flags,
                torch.tensor(self.tokenizer.eos_id, device=x.device),
                next_tokens
            )

            # FIX: only accumulate score for sequences that are still running
            scores = torch.where(finished_flags, scores, scores + token_scores)

            # FIX: only append tokens for unfinished sequences to avoid padding
            # sequences with repeated EOS tokens. We always append here for shape
            # consistency, but finished sequences get a no-op EOS that will be
            # stripped by post_process_sequence.
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)  # (batch_size, seq_len+1)

            is_eos = next_tokens == self.tokenizer.eos_id
            finished_flags = finished_flags | is_eos

        return x, scores

    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            beam_width: Number of beams to use
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, beam_width, sequence_length)
               where each sequence in a beam set is sorted by score (descending)
             - scores is of shape (batch_size, beam_width)
        """
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")

        batch_size, seq_len = x.shape
        device = x.device

        # Initialize beams by replicating input sequences
        # Shape: (batch_size, beam_width, seq_len)
        sequences = x.unsqueeze(1).expand(batch_size, beam_width, -1).clone()

        # Initialize scores — first beam is active, rest are masked out
        # Shape: (batch_size, beam_width)
        scores = torch.full((batch_size, beam_width), float('-inf'), device=device)
        scores[:, 0] = 0.0

        # Track which beams have finished (reached EOS)
        # Shape: (batch_size, beam_width)
        finished = torch.zeros((batch_size, beam_width), dtype=torch.bool, device=device)

        for step in range(self.max_length - seq_len):
            # -----------------------------------------------------------------
            # FIX: batch ALL beams into a single score_fn call instead of
            # calling it beam_width times.  This is the primary cause of the
            # ~beam_width× slowdown observed in validation.
            #
            # Reshape (batch_size, beam_width, seq_len) →
            #         (batch_size * beam_width, seq_len)
            # run one forward pass, then reshape back.
            # -----------------------------------------------------------------
            current_seq_len = sequences.size(-1)
            x_flat = sequences.view(batch_size * beam_width, current_seq_len)
            logits_flat = self.score_fn(x_flat)  # (batch_size*beam_width, vocab_size)
            vocab_size = logits_flat.size(-1)
            # Shape: (batch_size, beam_width, vocab_size)
            logits = logits_flat.view(batch_size, beam_width, vocab_size)

            # Apply temperature / top-k / top-p filtering
            filtered_logits = self._filter_logits(logits, temperature, top_k=0, top_p=1.0)

            # Apply repeat penalty using current sequences
            filtered_logits = self._apply_repeat_penalty(filtered_logits, sequences, repeat_penalty)

            # For finished beams only allow EOS (so their score stays stable)
            finished_mask = finished.unsqueeze(-1)  # (batch_size, beam_width, 1)
            filtered_logits = torch.where(
                finished_mask,
                torch.full_like(filtered_logits, float('-inf')),
                filtered_logits
            )
            # Allow EOS for finished beams with score 0 (no-op extension)
            eos_score = torch.where(
                finished,
                torch.zeros_like(scores),
                filtered_logits[:, :, self.tokenizer.eos_id]
            )
            filtered_logits[:, :, self.tokenizer.eos_id] = eos_score

            # Compute candidate scores: current beam scores + log-probs of next tokens
            # Shape: (batch_size, beam_width, vocab_size)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            candidate_scores = scores.unsqueeze(-1) + log_probs  # broadcast over vocab

            # Flatten to pick top beam_width candidates across all beams × vocab
            # Shape: (batch_size, beam_width * vocab_size)
            flat_scores = candidate_scores.reshape(batch_size, -1)

            # Shape: (batch_size, beam_width)
            top_scores, top_indices = torch.topk(flat_scores, beam_width, dim=-1)

            top_beam_idx  = top_indices // vocab_size  # which beam did this come from
            top_token_idx = top_indices  % vocab_size  # which token was selected

            # Gather parent sequences for each new beam
            batch_idx = torch.arange(batch_size, device=device).unsqueeze(-1)  # (batch_size, 1)
            new_sequences = sequences[batch_idx, top_beam_idx]  # (batch_size, beam_width, seq_len)

            # Append newly chosen tokens
            sequences = torch.cat(
                [new_sequences, top_token_idx.unsqueeze(-1)], dim=-1
            )  # (batch_size, beam_width, seq_len+1)

            scores = top_scores  # (batch_size, beam_width)

            # Propagate finished flags and mark newly completed beams
            is_eos = top_token_idx == self.tokenizer.eos_id
            prev_finished = finished[batch_idx, top_beam_idx]  # realign with reordered beams
            finished = prev_finished | is_eos

            if finished.all():
                break

        # Sort beams by score (descending) for each batch item
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(-1)
        sorted_sequences = sequences[batch_idx, sorted_indices]

        return sorted_sequences, sorted_scores

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            top_k: Number of top-k tokens to sample from
            top_p: Proportion of top-p tokens to sample from
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")

        batch_size = x.size(0)
        scores   = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break

            next_logits = self.score_fn(x)  # (batch_size, vocab_size)
            filtered_logits = self._filter_logits(next_logits, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)

            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (batch_size,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)

            # Only accumulate scores for sequences still running
            scores = torch.where(finished, scores, scores + token_scores)

            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

            is_eos = next_tokens == self.tokenizer.eos_id
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions
        Returns:
            if seq is a single sequence, return a tensor of same shape with sequence truncated at EOS
            if seq is a batch of sequences, return a list of tensors with each sequence truncated at first EOS
        """
        # Handle single sequence case
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq

        # Handle batched sequences
        eos_mask = seq == tokenizer.eos_id  # (batch_size, sequence_length)
        # Find first EOS token in each sequence
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        # Create sequence mask that includes everything up to and including first EOS
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]