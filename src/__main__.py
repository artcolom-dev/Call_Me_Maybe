# ABOUTME: Entry point for the Call Me Maybe function calling system.
# ABOUTME: Handles CLI arguments, orchestrates loading, generation, and output.


import argparse
import json
import sys
from pathlib import Path

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from typing import Any, Optional

from .constrained_decoder import ConstrainedDecoder
from .json_schema import build_template
from .models import FunctionCallResult, FunctionDefinition
from .parser import load_function_definitions, load_test_prompts
from .tokenizer import Tokenizer
from .visualization import GenerationVisualizer

DEFAULTS = {
    "functions": "data/input/functions_definition.json",
    "input": "data/input/function_calling_tests.json",
    "output": "data/output/function_calling_results.json",
    "model": "Qwen/Qwen3-0.6B",
}


def _fix_json_escapes(raw: str) -> str:
    """Fix invalid JSON escape sequences inside string values.

    Walks the raw JSON with a state machine to find string
    boundaries, then fixes backslashes inside strings:
    - \\' becomes ' (apostrophe needs no escaping in JSON)
    - \\X where X is not a valid JSON escape becomes \\\\X
    - \\uXXXX with non-hex digits becomes \\\\uXXXX
    - Valid escapes (\\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t,
      \\uHHHH) are kept as-is.

    Args:
        raw: Raw JSON string that may contain invalid escapes.

    Returns:
        JSON string with all escapes valid for json.loads.
    """
    result: list[str] = []
    in_string = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if not in_string:
            result.append(ch)
            if ch == '"':
                in_string = True
            i += 1
        elif ch == '"':
            result.append(ch)
            in_string = False
            i += 1
        elif ch == '\\' and i + 1 < len(raw):
            nxt = raw[i + 1]
            if nxt == "'":
                result.append("'")
                i += 2
            elif nxt in '"\\/':
                result.append(ch)
                result.append(nxt)
                i += 2
            elif nxt == 'u':
                hex_ok = (
                    i + 5 < len(raw)
                    and all(
                        c in '0123456789abcdefABCDEF'
                        for c in raw[i + 2:i + 6]
                    )
                )
                if hex_ok:
                    result.append(ch)
                    result.append(nxt)
                    i += 2
                else:
                    result.append('\\')
                    result.append('\\')
                    i += 1
            elif nxt in 'bfnrt':
                result.append(ch)
                result.append(nxt)
                i += 2
            else:
                result.append('\\')
                result.append('\\')
                i += 1
        else:
            result.append(ch)
            i += 1
    return ''.join(result)


def _coerce_param_types(
    params: dict[str, Any], fn: Optional[FunctionDefinition]
) -> dict[str, Any]:
    """Convert parameter values to match their declared types.

    Args:
        params: Raw parameters from JSON parsing.
        fn: The function definition (None if unknown function).

    Returns:
        Parameters with corrected types.
    """
    if fn is None:
        return params
    result: dict[str, Any] = {}
    for key, value in params.items():
        param_def = fn.parameters.get(key)
        if param_def and param_def.type in ("number", "float"):
            result[key] = float(value)
        elif param_def and param_def.type in ("integer", "int"):
            result[key] = int(value)
        elif param_def and param_def.type in ("boolean", "bool"):
            result[key] = bool(value)
        else:
            result[key] = value
    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls."
    )
    p.add_argument("--functions_definition", default=DEFAULTS["functions"])
    p.add_argument("--input", default=DEFAULTS["input"])
    p.add_argument("--output", default=DEFAULTS["output"])
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def write_results(results: list[FunctionCallResult], path: str) -> None:
    """Write function call results to a JSON file.

    Args:
        results: List of function call results.
        path: Output file path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.model_dump() for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Results written to {output_path}")


def main() -> None:
    """Main entry point for the function calling system."""
    args = parse_args()

    # Load input files
    print("Loading function definitions...")
    functions = load_function_definitions(args.functions_definition)
    print(f"  Loaded {len(functions)} function(s).")

    print("Loading test prompts...")
    prompts = load_test_prompts(args.input)
    print(f"  Loaded {len(prompts)} prompt(s).")

    print(f"\nLoading model {args.model}...")
    try:
        model = Small_LLM_Model(args.model)
    except Exception as e:
        print(f"Error: failed to load model '{args.model}': {e}",
              file=sys.stderr)
        sys.exit(1)

    print("Loading tokenizer...")
    try:
        tok = Tokenizer.from_model_files(
            model.get_path_to_vocab_file(),
            model.get_path_to_merges_file(),
        )
    except Exception as e:
        print(f"Error: failed to load tokenizer for '{args.model}': {e}",
              file=sys.stderr)
        sys.exit(1)

    templates = [build_template(fn) for fn in functions]
    vocab_size = len(tok.vocab)
    vis = GenerationVisualizer(vocab_size=vocab_size) if args.verbose else None
    decoder = ConstrainedDecoder(
        model=model, tokenizer=tok, templates=templates,
        visualizer=vis,
    )

    fn_by_name = {fn.name: fn for fn in functions}

    results: list[FunctionCallResult] = []
    print(f"\nProcessing {len(prompts)} prompt(s)...")
    for i, prompt in enumerate(prompts):
        print(f"  [{i + 1}/{len(prompts)}] {prompt.prompt}")
        raw_json = decoder.generate(prompt.prompt)
        raw_json = _fix_json_escapes(raw_json)
        try:
            parsed = json.loads(raw_json)
            params = _coerce_param_types(
                parsed["parameters"], fn_by_name.get(parsed["name"])
            )
            results.append(FunctionCallResult(
                prompt=prompt.prompt,
                name=parsed["name"],
                parameters=params,
            ))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    Error parsing output: {e}", file=sys.stderr)
            results.append(FunctionCallResult(
                prompt=prompt.prompt,
                name="error",
                parameters={},
            ))

    write_results(results, args.output)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
