# ABOUTME: Visualization of the constrained decoding generation process.
# ABOUTME: Shows token-by-token progress with logit scores and top candidates.

import numpy as np
from pydantic import BaseModel


class GenerationVisualizer(BaseModel):
    """Displays each generation step with scores and alternatives.

    Attributes:
        top_k: Number of top candidates to show at each step.
        vocab_size: Total vocabulary size for context.
    """

    top_k: int = 5
    vocab_size: int = 0

    def show_header(self, prompt: str) -> None:
        """Print the header for a new generation.

        Args:
            prompt: The user query being processed.
        """
        print(f"\n  Generating for: {prompt}")
        print(f"  Vocabulary: {self.vocab_size} tokens\n")

    def show_step(
        self,
        step: int,
        chosen_id: int,
        logits: np.ndarray,
        valid_ids: list[int],
        inverse_vocab: dict[int, str],
        buffer: str,
    ) -> None:
        """Display one generation step.

        Args:
            step: The current step number.
            chosen_id: The token ID that was selected.
            logits: The masked logits array.
            valid_ids: List of valid token IDs at this step.
            inverse_vocab: Mapping from token ID to token string.
            buffer: The JSON text generated so far.
        """
        scores: list[tuple[int, float]] = []
        for tid in valid_ids:
            score = float(logits[tid])
            if not np.isinf(score):
                scores.append((tid, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:self.top_k]
        if not top:
            return

        n_valid = len(scores)
        chosen_text = inverse_vocab.get(chosen_id, "?").replace("Ġ", " ")
        print(
            f"  step {step:<3}  "
            f"{repr(chosen_text):<16}  "
            f"({n_valid} valid tokens)"
        )
        for rank, (tid, score) in enumerate(top):
            token_raw = inverse_vocab.get(tid, "?")
            token_display = repr(token_raw.replace("Ġ", " "))
            marker = "  <-" if tid == chosen_id else ""
            print(
                f"           {rank + 1}. "
                f"{token_display:<16} {score:>8.2f}{marker}"
            )
        print()

    def show_result(self, buffer: str, total_steps: int) -> None:
        """Print the final generated JSON.

        Args:
            buffer: The complete generated JSON string.
            total_steps: Number of generation steps taken.
        """
        print(f"  Result: {buffer}")
        print(f"  Done in {total_steps} steps\n")
