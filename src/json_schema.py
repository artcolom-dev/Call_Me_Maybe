# ABOUTME: Template-based JSON validation for constrained decoding.
# ABOUTME: Builds expected JSON templates and checks if generated text matches.

from pydantic import BaseModel

from .models import FunctionDefinition


class TemplateSegment(BaseModel):
    """A piece of a JSON template.

    Attributes:
        text: The fixed text (if fixed segment) or empty string.
        is_variable: Whether this is a variable part (number/string).
        var_type: The type of variable ("number" or "string").
    """

    text: str = ""
    is_variable: bool = False
    var_type: str = ""


class FunctionTemplate(BaseModel):
    """A JSON template for one function.

    Attributes:
        fn: The function definition this template is for.
        segments: The ordered list of fixed and variable parts.
    """

    fn: FunctionDefinition
    segments: list[TemplateSegment]


def build_template(fn: FunctionDefinition) -> FunctionTemplate:
    """Build a JSON template for a function definition.

    Args:
        fn: The function definition.

    Returns:
        A FunctionTemplate with all segments.
    """
    segments: list[TemplateSegment] = []
    header = '{"name": "' + fn.name + '", "parameters": {'
    params = list(fn.parameters.items())
    for i, (pname, pparam) in enumerate(params):
        header += '"' + pname + '": '
        segments.append(TemplateSegment(text=header))
        header = ""
        if pparam.type in ("string", "str"):
            segments.append(TemplateSegment(text='"'))
            segments.append(TemplateSegment(
                is_variable=True, var_type="string"
            ))
            segments.append(TemplateSegment(text='"'))
        elif pparam.type in ("boolean", "bool"):
            segments.append(TemplateSegment(
                is_variable=True, var_type="boolean"
            ))
        elif pparam.type in ("integer", "int"):
            segments.append(TemplateSegment(
                is_variable=True, var_type="integer"
            ))
        else:
            segments.append(TemplateSegment(
                is_variable=True, var_type="number"
            ))
        if i < len(params) - 1:
            header = ", "
        else:
            header = "}}"
    segments.append(TemplateSegment(text=header))
    return FunctionTemplate(fn=fn, segments=segments)


def _is_valid_number_char(ch: str) -> bool:
    """Check if a character can appear in a JSON number."""
    return ch in "0123456789.-+eE"


def _is_valid_integer_char(ch: str) -> bool:
    """Check if a character can appear in a JSON integer."""
    return ch in "0123456789-"


def _is_bool_prefix_match(
    buffer: str, pos: int, valid: str, next_seg: str
) -> bool:
    """Check if buffer at pos is a valid prefix of a boolean value."""
    remaining_buf = buffer[pos:]
    if valid.startswith(remaining_buf):
        return True
    if remaining_buf.startswith(valid):
        after = pos + len(valid)
        if after >= len(buffer):
            return True
        if next_seg and _starts_with_at(buffer, after, next_seg):
            return True
    return False


def is_prefix_valid(buffer: str, template: FunctionTemplate) -> bool:
    """Check if buffer is a valid prefix of the template.

    Args:
        buffer: The text generated so far.
        template: The template to check against.

    Returns:
        True if buffer matches the beginning of this template.
    """
    pos = 0
    for seg_idx, seg in enumerate(template.segments):
        if pos >= len(buffer):
            return True
        if not seg.is_variable:
            for ch in seg.text:
                if pos >= len(buffer):
                    return True
                if buffer[pos] != ch:
                    return False
                pos += 1
        else:
            next_seg = _get_next_fixed(template, seg_idx)
            if seg.var_type == "number":
                while pos < len(buffer):
                    if next_seg and _starts_with_at(buffer, pos, next_seg):
                        break
                    if not _is_valid_number_char(buffer[pos]):
                        remaining = buffer[pos:]
                        if next_seg and next_seg.startswith(remaining):
                            return True
                        return False
                    pos += 1
            elif seg.var_type == "integer":
                while pos < len(buffer):
                    if next_seg and _starts_with_at(buffer, pos, next_seg):
                        break
                    if not _is_valid_integer_char(buffer[pos]):
                        remaining = buffer[pos:]
                        if next_seg and next_seg.startswith(remaining):
                            return True
                        return False
                    pos += 1
            elif seg.var_type == "boolean":
                for valid in ("true", "false"):
                    if _is_bool_prefix_match(
                        buffer, pos, valid, next_seg
                    ):
                        return True
                return False
            elif seg.var_type == "string":
                while pos < len(buffer):
                    if buffer[pos] == '\\':
                        pos += 2
                        continue
                    if next_seg and _starts_with_at(buffer, pos, next_seg):
                        break
                    pos += 1
            if pos >= len(buffer) and next_seg:
                return True
    return pos == len(buffer)


def _get_next_fixed(
    template: FunctionTemplate, current_idx: int
) -> str:
    """Get the text of the next fixed segment after current_idx."""
    for i in range(current_idx + 1, len(template.segments)):
        if not template.segments[i].is_variable:
            return template.segments[i].text
    return ""


def _starts_with_at(buffer: str, pos: int, text: str) -> bool:
    """Check if buffer at position pos starts with text."""
    return buffer[pos:pos + len(text)] == text
