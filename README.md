*This project has been created as part of the 42 curriculum by artcolom.*

# Call Me Maybe

## Description

A function calling system that translates natural language prompts into structured JSON function calls using **constrained decoding** with Qwen3-0.6B (0.6B parameters).

The core idea: instead of hoping a small LLM outputs valid JSON, we **force it** by masking invalid tokens at every generation step. The model can only pick tokens that produce a valid prefix of the expected JSON schema. This guarantees 100% parseable output regardless of model size.

```
"What is the product of 3 and 5?"
        |
        v
   [Qwen3-0.6B]  <-- only valid JSON tokens allowed at each step
        |
        v
{"name": "fn_multiply_numbers", "parameters": {"a": 3.0, "b": 5.0}}
```

## Algorithm Explanation

### Template-based constrained decoding

For each function definition, we build a **JSON template** -- an ordered list of segments that alternate between fixed text and variable slots:

```
FIXED:    {"name": "fn_multiply_numbers", "parameters": {"a":
VARIABLE: <number>
FIXED:    , "b":
VARIABLE: <number>
FIXED:    }}
```

At each generation step:

1. The LLM produces **logits** (probability scores) for every token in its ~150k vocabulary
2. We determine which tokens are **valid** -- appending them to the current buffer must still be a valid prefix of at least one template
3. All invalid tokens get their logits set to **-inf**
4. We pick the token with the **highest remaining logit** (greedy decoding)
5. Append it to the buffer and repeat until the JSON is complete (`}}`)

### Prefix validation

The function `is_prefix_valid(buffer, template)` walks through the template segments and checks that the buffer matches:
- **Fixed segments**: character-by-character exact match
- **Variable segments**: type-specific validation
  - `number`: digits, `.`, `-`, `+`, `e`, `E`
  - `integer`: digits and `-` only
  - `boolean`: must be a prefix of `true` or `false`
  - `string`: any character, with `\` skipping the next char (escape handling)

The boundary between a variable segment and the next fixed segment is detected by checking if the current buffer position starts matching the next fixed text.

### Optimization

Testing all ~150k tokens at each step would be slow. We use two optimizations:
- **First-character filtering**: we pre-compute a mapping from each ASCII character to all tokens starting with that character. At each step, we first test which characters are valid (~95 checks), then only test the tokens starting with those characters
- **Buffer caching**: if the same buffer state has been seen before, we reuse the cached valid token list

## Design Decisions

- **Template system over grammar**: instead of a full JSON grammar, we use function-specific templates. This is simpler, faster, and sufficient since the output schema is known in advance.
- **Greedy decoding**: we always pick the highest-logit valid token (argmax). No sampling, no beam search. This is deterministic and fast -- the constrained masking already ensures valid output.
- **Peek logic for embedded quotes**: when the model generates `"` inside a string variable, we can't tell if it's a closing quote or part of the text (e.g., `Say "hello"`). We peek at the next token: if the model would follow with `}` or `,`, it's a closing quote. Otherwise, we write `\"` (escaped quote) into the buffer.
- **Post-processing state machine**: the raw buffer may contain invalid JSON escapes (e.g., `\U` from Windows paths, `\'` from apostrophes). A state machine walks the JSON string values and fixes these: doubling backslashes for invalid escapes, stripping unnecessary `\'`.
- **Type coercion**: JSON parsing gives Python types that may not match the function definition (e.g., `int` instead of `float`). A post-processing step coerces values to their declared types.
- **Custom BPE tokenizer**: built from the model's `vocab.json` and `merges.txt` files, implementing pre-tokenization (space -> `Ġ`), iterative BPE merging, and ID lookup.

## Performance Analysis

- **Accuracy**: 11/11 (100%) on private moulinette tests, covering numbers, strings, integers, booleans, multi-parameter functions, Windows paths with backslashes, embedded quotes, and apostrophes
- **JSON validity**: 100% -- constrained decoding guarantees every output is parseable
- **Speed**: all 11 prompts processed in under 2 minutes on a MacBook (M-series). The first prompt is slowest due to model loading; subsequent prompts benefit from token caching
- **Robustness**: graceful error handling for missing files, invalid JSON input, and model loading failures

## Challenges Faced

1. **Type support**: the initial implementation only handled `number` and `string`. Adding `integer` and `boolean` required extending both the template builder and the prefix validator with type-specific logic.
2. **Garbage tokens**: some vocabulary tokens contain non-ASCII characters (e.g., `âĢĿ`) that leaked into string values. Fixed by filtering tokens with a `_is_clean_token()` check during initialization.
3. **Embedded quotes**: prompts like `Say "hello" to {name}` caused the decoder to prematurely close the string. The peek logic (look at what the model wants to generate after `"`) solved this.
4. **Backslash handling**: Windows paths like `C:\Users\john\config.ini` produce invalid JSON escapes (`\U`, `\j`, `\c`). A regex-based fix wasn't enough because `\u` followed by non-hex digits was treated as a valid Unicode escape prefix. Replacing the regex with a state machine that processes each backslash in context solved all edge cases.
5. **Apostrophe escaping**: the model generates `\'` for apostrophes, but JSON doesn't recognize this escape. The state machine converts `\'` to plain `'`.

## Testing Strategy

- **Moulinette grading**: used the provided moulinette to run the full test suite (11 private tests) and validate output format, function selection, and parameter extraction
- **Verbose mode**: the `--verbose` flag displays each generation step (chosen token, score, top-5 alternatives, valid token count) for manual inspection of the decoding process
- **Incremental testing**: each new feature (type support, escape handling, peek logic) was tested individually against the moulinette to ensure no regressions
- **Edge cases covered**: float vs int types, boolean values, multi-parameter functions, special characters in strings (quotes, apostrophes, backslashes), long numbers with decimals

## Instructions

### Install

```bash
make install
```

### Run

```bash
make run
```

With custom paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Verbose mode

```bash
uv run python -m src --verbose
```

### Lint

```bash
make lint
```

### Debug

```bash
make debug
```

## Example Usage

Given this function definition:
```json
{
  "name": "fn_multiply_numbers",
  "description": "Multiply two numbers together.",
  "parameters": { "a": {"type": "number"}, "b": {"type": "number"} },
  "returns": {"type": "number"}
}
```

And this prompt:
```json
{"prompt": "What is the product of 3 and 5?"}
```

The program outputs:
```json
{
  "prompt": "What is the product of 3 and 5?",
  "name": "fn_multiply_numbers",
  "parameters": {"a": 3.0, "b": 5.0}
}
```

## Resources

- [Constrained Decoding for LLMs](https://arxiv.org/abs/2307.09702) -- Guided Generation of Large Language Models (original paper)
- [BPE Tokenization](https://huggingface.co/learn/nlp-course/en/chapter6/5) -- Hugging Face NLP course on Byte-Pair Encoding
- [JSON specification](https://www.json.org/json-en.html) -- Official JSON format reference
- [Pydantic documentation](https://docs.pydantic.dev/) -- Data validation library used throughout

### AI Usage

Claude (Anthropic) was used as a research and debugging assistant during development:
- **Research**: understanding BPE tokenization internals, JSON escape sequence edge cases, constrained decoding strategies, and how logit masking works in practice
- **Debugging**: extensive iterative debugging sessions to diagnose template matching issues, escape handling edge cases, type coercion problems, and embedded quote detection across many test runs
- **No blind generation**: all architectural decisions (template system, peek logic, state machine for escapes) were discussed, understood, and validated through testing. The core design and implementation choices are my own.
