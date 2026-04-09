# ABOUTME: Custom tokenizer built from the model's vocabulary and merges files.
# ABOUTME: Reimplements encode/decode without using the SDK's built-in methods.

import json

from pydantic import BaseModel


class Tokenizer(BaseModel):
    """Custom tokenizer that loads vocabulary from the model files.

    Attributes:
        vocab: Mapping of token strings to their integer IDs.
        inverse_vocab: Mapping of integer IDs to their token strings.
    """

    vocab: dict[str, int]
    inverse_vocab: dict[int, str]
    merges: dict[tuple[str, str], int]

    @classmethod
    def from_model_files(
        cls, vocab_path: str, merges_path: str
    ) -> "Tokenizer":
        """Build a Tokenizer from vocab.json and merges.txt.

        Args:
            vocab_path: Path to the vocabulary JSON file.
            merges_path: Path to the BPE merges text file.

        Returns:
            A Tokenizer instance ready to encode and decode.
        """
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        inverse_vocab = {token_id: token_str
                         for token_str, token_id in vocab.items()}
        merges: dict[tuple[str, str], int] = {}
        with open(merges_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(" ")
                if len(parts) == 2:
                    merges[(parts[0], parts[1])] = i
        return cls(
            vocab=vocab, inverse_vocab=inverse_vocab, merges=merges
        )

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a text string.

        Each ID is looked up in the inverse vocabulary to get its
        token string. The special character 'Ġ' is replaced by a
        space. Leading spaces are stripped.

        Args:
            ids: List of token IDs to decode.

        Returns:
            The decoded text string.
        """
        parts: list[str] = []
        for token_id in ids:
            token_str = self.inverse_vocab.get(token_id, "")
            parts.append(token_str)
        text = "".join(parts)
        text = text.replace("Ġ", " ")
        # First token often has a leading Ġ that produces an extra space
        text = text.lstrip(" ")
        return text

    def _pre_tokenize(self, text: str) -> list[str]:
        """Split text into individual characters with space markers.

        Spaces are replaced by 'Ġ' and attached to the following
        character. This matches how the BPE vocabulary represents
        word boundaries.

        Args:
            text: The input text to pre-tokenize.

        Returns:
            List of single characters (with 'Ġ' replacing spaces).
        """
        chars: list[str] = []
        for ch in text:
            if ch == " ":
                chars.append("Ġ")
            else:
                chars.append(ch)
        return chars

    def _bpe_merge(self, tokens: list[str]) -> list[str]:
        """Apply BPE merges repeatedly until no more merges apply.

        At each step, find the pair of adjacent tokens with the
        highest priority (lowest index in merges.txt), merge them
        into one token, and repeat.

        Args:
            tokens: List of token strings to merge.

        Returns:
            List of merged token strings.
        """
        while len(tokens) > 1:
            # Find the pair with the highest priority (lowest rank)
            best_pair = None
            best_rank = float("inf")
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merges.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            # No more merges possible
            if best_pair is None:
                break
            # Merge all occurrences of this pair
            merged = best_pair[0] + best_pair[1]
            new_tokens: list[str] = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1
                        and tokens[i] == best_pair[0]
                        and tokens[i + 1] == best_pair[1]):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode a text string into a list of token IDs.

        Steps:
        1. Pre-tokenize: split into characters with space markers.
        2. BPE merge: repeatedly merge pairs using merges.txt rules.
        3. Lookup: convert each final token to its ID via vocab.

        Args:
            text: The text to encode.

        Returns:
            List of integer token IDs.
        """
        chars = self._pre_tokenize(text)
        tokens = self._bpe_merge(chars)
        ids: list[int] = []
        for token in tokens:
            token_id = self.vocab.get(token)
            if token_id is not None:
                ids.append(token_id)
        return ids
