# ABOUTME: Constrained decoder that generates valid JSON token by token.
# ABOUTME: Filters LLM logits to only allow tokens matching the JSON template.

import numpy as np
from pydantic import BaseModel, ConfigDict

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from typing import Optional

from .json_schema import (
    FunctionTemplate, is_prefix_valid, _get_next_fixed,
)
from .tokenizer import Tokenizer
from .visualization import GenerationVisualizer


def _is_clean_token(text: str) -> bool:
    """Check if a token only contains usable characters."""
    return all(32 <= ord(c) < 127 or c == '\n' or c == '\t' for c in text)


class ConstrainedDecoder(BaseModel):
    """Generates JSON by filtering LLM output at each token.

    Attributes:
        model: The LLM model used for inference.
        tokenizer: Custom tokenizer for encode/decode.
        templates: One template per available function.
        max_tokens: Maximum tokens to generate before stopping.
        char_to_tokens: Pre-computed mapping of first char to token IDs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    tokenizer: Tokenizer
    templates: list[FunctionTemplate]
    max_tokens: int = 200
    char_to_tokens: dict[str, list[int]] = {}
    _valid_tokens_cache: dict[str, list[int]] = {}
    visualizer: Optional[GenerationVisualizer] = None

    def model_post_init(self, __context: object) -> None:
        """Pre-compute the first-character to token IDs mapping."""
        mapping: dict[str, list[int]] = {}
        for token_id, token_text in self.tokenizer.inverse_vocab.items():
            clean = token_text.replace("Ġ", " ")
            if not clean or not _is_clean_token(clean):
                continue
            ch = clean[0]
            if ch not in mapping:
                mapping[ch] = []
            mapping[ch].append(token_id)
        self.char_to_tokens = mapping
        self._valid_tokens_cache = {}

    def generate(self, prompt: str) -> str:
        """Generate constrained JSON from a natural language prompt.

        Args:
            prompt: The user query.

        Returns:
            The generated JSON string.
        """
        full_prompt = self._build_prompt(prompt)
        input_ids = self.tokenizer.encode(full_prompt)
        buffer = ""

        if self.visualizer:
            self.visualizer.show_header(prompt)

        for step in range(self.max_tokens):
            logits = self.model.get_logits_from_input_ids(input_ids)
            logits_arr = np.array(logits)
            valid_ids = self._get_valid_token_ids(buffer)
            self._mask_invalid_tokens(logits_arr, buffer)
            if np.all(np.isinf(logits_arr)):
                break
            next_id = int(np.argmax(logits_arr))

            if self.visualizer:
                self.visualizer.show_step(
                    step + 1, next_id, logits_arr,
                    valid_ids, self.tokenizer.inverse_vocab,
                    buffer,
                )

            token_text = self.tokenizer.inverse_vocab.get(next_id, "")
            clean_text = token_text.replace("Ġ", " ")

            if (clean_text == '"'
                    and self._in_string_position(buffer)):
                peek_ids = input_ids + [next_id]
                peek_logits = self.model.get_logits_from_input_ids(
                    peek_ids
                )
                peek_id = int(np.argmax(np.array(peek_logits)))
                peek_text = self.tokenizer.inverse_vocab.get(
                    peek_id, ""
                ).replace("Ġ", " ")
                if not peek_text or peek_text.lstrip()[:1] in ('}', ','):
                    buffer += '"'
                else:
                    buffer += '\\"'
            else:
                buffer += clean_text

            input_ids.append(next_id)
            if self._is_complete(buffer):
                break

        if self.visualizer:
            self.visualizer.show_result(buffer.strip(), step + 1)

        return buffer.strip()

    def _in_string_position(self, buffer: str) -> bool:
        """Check if the buffer is currently inside a string variable."""
        for tmpl in self.templates:
            pos = 0
            for seg_idx, seg in enumerate(tmpl.segments):
                if pos > len(buffer):
                    break
                if not seg.is_variable:
                    match = True
                    for ch in seg.text:
                        if pos >= len(buffer):
                            match = False
                            break
                        if buffer[pos] != ch:
                            match = False
                            break
                        pos += 1
                    if not match:
                        break
                else:
                    next_fixed = _get_next_fixed(tmpl, seg_idx)
                    if seg.var_type == "string":
                        while pos < len(buffer):
                            if buffer[pos] == '\\':
                                pos += 2
                                continue
                            if (next_fixed
                                    and buffer[pos:pos + len(next_fixed)]
                                    == next_fixed):
                                break
                            pos += 1
                        if pos >= len(buffer):
                            return True
                    else:
                        while pos < len(buffer):
                            if (next_fixed
                                    and buffer[pos:pos + len(next_fixed)]
                                    == next_fixed):
                                break
                            pos += 1
        return False

    def _build_prompt(self, prompt: str) -> str:
        """Build the full prompt with function context.

        Args:
            prompt: The user query.

        Returns:
            Formatted prompt string for the LLM.
        """
        fn_lines = []
        for tmpl in self.templates:
            fn = tmpl.fn
            param_parts = []
            for k, v in fn.parameters.items():
                param_parts.append(f"{k}: {v.type}")
            params = ", ".join(param_parts)
            fn_lines.append(f"- {fn.name}({params}): {fn.description}")
        functions_block = "\n".join(fn_lines)
        return (
            "You are a function calling assistant.\n"
            f"Available functions:\n{functions_block}\n\n"
            "Reply with JSON: "
            '{"name": "<function>", "parameters": {<args>}}\n'
            "Rules: Copy values EXACTLY from the user message. "
            "number=>float, integer=>int, string=>str, "
            "boolean=>bool.\n\n"
            'User: "Add 5 and 3"\n'
            'Assistant: {"name": "fn_add_numbers", '
            '"parameters": {"a": 5.0, "b": 3.0}}\n\n'
            f'User: "{prompt}"\nAssistant: '
        )

    def _get_valid_first_chars(self, buffer: str) -> set[str]:
        """Determine which first characters are valid for the next token.

        Tests each possible first character against is_prefix_valid.
        Simple and reliable, ~200 checks per template.

        Args:
            buffer: The JSON text generated so far.

        Returns:
            Set of characters that could start the next token.
        """
        valid_chars: set[str] = set()
        for ch in self.char_to_tokens:
            candidate = buffer + ch
            for tmpl in self.templates:
                if is_prefix_valid(candidate, tmpl):
                    valid_chars.add(ch)
                    break
        return valid_chars

    def _get_valid_token_ids(self, buffer: str) -> list[int]:
        """Get the list of valid token IDs for the current buffer.

        Uses a cache: if the same buffer was seen before (e.g.
        across different prompts), the result is reused instantly.

        Args:
            buffer: The JSON text generated so far.

        Returns:
            List of token IDs that produce valid continuations.
        """
        if buffer in self._valid_tokens_cache:
            return self._valid_tokens_cache[buffer]
        valid_chars = self._get_valid_first_chars(buffer)
        valid_ids: list[int] = []
        for ch in valid_chars:
            token_ids = self.char_to_tokens.get(ch, [])
            for token_id in token_ids:
                token_text = self.tokenizer.inverse_vocab[token_id]
                clean_text = token_text.replace("Ġ", " ")
                candidate = buffer + clean_text
                for tmpl in self.templates:
                    if is_prefix_valid(candidate, tmpl):
                        valid_ids.append(token_id)
                        break
        self._valid_tokens_cache[buffer] = valid_ids
        return valid_ids

    def _mask_invalid_tokens(
        self, logits: np.ndarray, buffer: str
    ) -> None:
        """Set logits of invalid tokens to -inf.

        Uses cached valid token IDs to avoid redundant computation.

        Args:
            logits: Array of logits to modify in place.
            buffer: The JSON text generated so far.
        """
        valid_ids = self._get_valid_token_ids(buffer)
        original = logits.copy()
        logits[:] = -np.inf
        for token_id in valid_ids:
            logits[token_id] = original[token_id]

    def _is_complete(self, buffer: str) -> bool:
        """Check if the buffer is a complete valid JSON object."""
        return buffer.endswith("}}")
