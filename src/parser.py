# ABOUTME: Input file loading and validation with graceful error handling.
# ABOUTME: Reads function definitions and test prompts from JSON files.

import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .models import FunctionDefinition, TestPrompt


def load_json_file(path: str) -> list[dict]:
    """Load and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content as a list of dicts.

    Raises:
        SystemExit: If the file is missing or contains invalid JSON.
    """
    file_path = Path(path)
    if not file_path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print(f"Error: expected a JSON array in {path}", file=sys.stderr)
        sys.exit(1)
    return data


def load_function_definitions(path: str) -> list[FunctionDefinition]:
    """Load function definitions from a JSON file.

    Args:
        path: Path to the function definitions file.

    Returns:
        List of validated FunctionDefinition objects.
    """
    raw = load_json_file(path)
    definitions: list[FunctionDefinition] = []
    for i, item in enumerate(raw):
        try:
            definitions.append(FunctionDefinition(**item))
        except ValidationError as e:
            print(
                f"Error: invalid function definition at index {i}: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
    if not definitions:
        print(f"Error: no function definitions found in {path}",
              file=sys.stderr)
        sys.exit(1)
    return definitions


def load_test_prompts(path: str) -> list[TestPrompt]:
    """Load test prompts from a JSON file.

    Args:
        path: Path to the test prompts file.

    Returns:
        List of validated TestPrompt objects.
    """
    raw = load_json_file(path)
    prompts: list[TestPrompt] = []
    for i, item in enumerate(raw):
        try:
            prompts.append(TestPrompt(**item))
        except ValidationError as e:
            print(
                f"Error: invalid test prompt at index {i}: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
    if not prompts:
        print(f"Error: no test prompts found in {path}",
              file=sys.stderr)
        sys.exit(1)
    return prompts
