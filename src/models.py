# ABOUTME: Pydantic models for function definitions, test prompts, and results.
# ABOUTME: All data structures used throughout the project are defined here.

from typing import Any
from pydantic import BaseModel


class FunctionParameter(BaseModel):
    """Schema for a single function parameter.

    Attributes:
        type: The expected type of the parameter (e.g. "number", "string").
    """

    type: str


class FunctionDefinition(BaseModel):
    """Schema for a callable function definition.

    Attributes:
        name: The function identifier (e.g. "fn_add_numbers").
        description: Human-readable description of what the function does.
        parameters: Mapping of parameter names to their type schemas.
        returns: The return type schema of the function.
    """

    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: FunctionParameter


class TestPrompt(BaseModel):
    """A single natural-language test prompt.

    Attributes:
        prompt: The user query to be translated into a function call.
    """

    prompt: str


class FunctionCallResult(BaseModel):
    """Result of translating a prompt into a function call.

    Attributes:
        prompt: The original natural-language request.
        name: The name of the selected function.
        parameters: The extracted arguments matching the function schema.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]
