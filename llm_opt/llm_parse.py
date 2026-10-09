import json
from typing import Any, Optional, Tuple
import re


def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing Markdown code fences like ```json ... ``` or ``` ... ```.

    Returns a trimmed string without surrounding fences; leaves inner backticks untouched.
    """
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def parse_json(text: str) -> Tuple[str, Optional[Any]]:
    """Clean an LLM response and parse JSON if possible.

    Returns (cleaned_text, parsed_obj or None).
    """
    cleaned = _strip_code_fences(text)
    try:
        return cleaned, json.loads(cleaned)
    except Exception:
        return cleaned, None


def fix_common_json_issues(text: str) -> str:
    """Attempt to fix common JSON issues:
    - Remove trailing commas before '}' or ']'
    - Remove invalid control characters (keep \n, \r, \t)
    - Trim BOM and surrounding spaces
    """
    if not isinstance(text, str):
        return text
    s = text
    # Strip BOM
    s = s.lstrip("\ufeff")
    # Remove trailing commas: ,\s*] or ,\s*}
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    # Remove non-printable control chars except \n, \r, \t
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s)
    return s.strip()


def normalize_json_newlines(text: str) -> str:
    """Normalize line breaks to improve readability and reduce parse errors from cramped JSON."""
    if not isinstance(text, str):
        return text
    s = text
    s = s.replace('},', '},\n')
    s = s.replace('],', '],\n')
    s = re.sub(r"}\s*{", "},\n{", s)
    s = re.sub(r",\s*(?=[{\"])", ",\n", s)
    return s


def extract_json_array(text: str) -> Tuple[str, Optional[Any]]:
    """Heuristically extract the largest JSON array from mixed text and parse it."""
    if not isinstance(text, str):
        return text, None
    s = text
    start = s.find('[')
    end = s.rfind(']')
    if start == -1 or end == -1 or end <= start:
        return s, None
    candidate = s[start:end+1]
    candidate = fix_common_json_issues(candidate)
    candidate = normalize_json_newlines(candidate)
    try:
        return candidate, json.loads(candidate)
    except Exception:
        for _ in range(3):
            end = candidate.rfind(']')
            if end == -1:
                break
            candidate = candidate[:end+1]
            try:
                fixed = normalize_json_newlines(fix_common_json_issues(candidate))
                return fixed, json.loads(fixed)
            except Exception:
                continue
    return candidate, None


def obtain_valid_json_from_llm(prompt: str) -> Tuple[str, Optional[Any]]:
    """Obtain a valid JSON string from LLM given a prompt, with fallbacks."""
    try:
        from llm_opt.llm_api import chat_with_llm, chat_with_llm_json  # type: ignore
    except Exception:  # pragma: no cover
        from llm_opt.llm_api import chat_with_llm  # type: ignore
        chat_with_llm_json = None  # type: ignore

    raw = chat_with_llm(prompt)
    cleaned, parsed = parse_json(raw)
    if parsed is not None:
        return cleaned, parsed
    ext_cleaned, ext_parsed = extract_json_array(raw)
    if ext_parsed is not None:
        return ext_cleaned, ext_parsed
    fixed = normalize_json_newlines(fix_common_json_issues(raw))
    fixed_cleaned, fixed_parsed = parse_json(fixed)
    if fixed_parsed is not None:
        return fixed_cleaned, fixed_parsed

    if chat_with_llm_json is not None:
        raw2 = chat_with_llm_json(prompt)
        cleaned2, parsed2 = parse_json(raw2)
        if parsed2 is not None:
            return cleaned2, parsed2
        ext2_cleaned, ext2_parsed = extract_json_array(raw2)
        if ext2_parsed is not None:
            return ext2_cleaned, ext2_parsed
        fixed2 = normalize_json_newlines(fix_common_json_issues(raw2))
        cleaned2b, parsed2b = parse_json(fixed2)
        if parsed2b is not None:
            return cleaned2b, parsed2b

    final = normalize_json_newlines(fix_common_json_issues(raw))
    return parse_json(final)


# ===================== Enhanced tolerant parsing helpers =====================

def _extract_json_payload_balanced(text: str) -> Tuple[str, Optional[Any]]:
    """Extract the first balanced JSON array or object from mixed text.

    This scanner is string-aware and handles nested brackets. Returns (snippet, obj or None).
    """
    if not isinstance(text, str):
        return text, None

    def scan_for(open_ch: str, close_ch: str) -> Tuple[str, Optional[Any]]:
        start = text.find(open_ch)
        if start == -1:
            return text, None
        balance = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                elif ch == open_ch:
                    balance += 1
                elif ch == close_ch:
                    balance -= 1
                    if balance == 0:
                        snippet = text[start:i+1]
                        try:
                            return snippet, json.loads(snippet)
                        except Exception:
                            break
        return text, None

    # Prefer array
    snip, obj = scan_for('[', ']')
    if obj is not None:
        return snip, obj
    # Fallback to object
    snip, obj = scan_for('{', '}')
    return snip, obj


def _normalize_factors_structure(data: Any) -> Any:
    """Normalize LLM output structure:
    - Allow top-level list or {"factors": [...]}
    - Ensure each item has an "info" dict
    - Hoist root-level "name"/"desc" into info if present
    Returns list for factors; if input is dict with other fields, returns dict with normalized factors.
    """
    def normalize_list(lst):
        out = []
        for item in lst or []:
            if not isinstance(item, dict):
                continue
            it = dict(item)
            info = dict(it.get('info') or {})
            if 'name' in it and 'name' not in info:
                info['name'] = it.pop('name')
            if 'desc' in it and 'desc' not in info:
                info['desc'] = it.pop('desc')
            it['info'] = info
            if 'idx' in it and 'expr' in it:
                out.append(it)
        return out

    if isinstance(data, list):
        return normalize_list(data)
    if isinstance(data, dict):
        if 'factors' in data and isinstance(data['factors'], list):
            data = dict(data)
            data['factors'] = normalize_list(data['factors'])
            return data
    return data


def load_json_relaxed(text: str) -> Tuple[str, Optional[Any]]:
    """Public helper to parse potentially noisy LLM output into JSON with structure normalization.

    Returns (snippet_text, parsed_obj or None).
    """
    # 1) Fast path
    cleaned, parsed = parse_json(text)
    if parsed is not None:
        return cleaned, _normalize_factors_structure(parsed)

    # 2) Extract largest array
    arr_text, arr_obj = extract_json_array(text)
    if arr_obj is not None:
        return arr_text, _normalize_factors_structure(arr_obj)

    # 3) Balanced scan for array/object
    snip, obj = _extract_json_payload_balanced(text)
    if obj is not None:
        return snip, _normalize_factors_structure(obj)

    # 4) Final attempt: clean and parse
    final = normalize_json_newlines(fix_common_json_issues(text))
    cleaned2, parsed2 = parse_json(final)
    if parsed2 is not None:
        return cleaned2, _normalize_factors_structure(parsed2)
    return final, None
