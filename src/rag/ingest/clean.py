"""Convert Hugo markup to searchable text while preserving fenced code."""

import re


def _attribute(tag: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*([\"'])(.*?)\1", tag, re.DOTALL)
    return match.group(2) if match else ""


def _clean_markup(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(
        r"{{<\s*glossary_tooltip\b.*?>}}",
        lambda m: _attribute(m[0], "text"),
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"{{<\s*(note|caution|warning)\s*>}}(.*?){{<\s*/\1\s*>}}",
        lambda m: f"{m[1].capitalize()}: {m[2].strip()}",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"""{{%\s*heading\s+(["'])(.*?)\1\s*%}}""",
        lambda m: "## " + m[2].title(),
        text,
    )
    text = re.sub(
        r"{{<\s*feature-state\b.*?>}}",
        lambda m: (
            f"Feature state: {_attribute(m[0], 'state')}"
            f" since {_attribute(m[0], 'for_k8s_version')}"
        ),
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"{{%\s*code_sample\b.*?%}}",
        lambda m: f"(Example file: {_attribute(m[0], 'file')})",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"{{%\s*tab\b(.*?)%}}(.*?){{%\s*/tab\s*%}}",
        lambda m: f"{_attribute(m[1], 'name')}: {m[2].strip()}",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"{{<.*?>}}|{{%.*?%}}", "", text, flags=re.DOTALL)
    # Images must be removed before links are flattened. Support nested URL parentheses.
    target = r"\((?:[^()\n]|\([^()\n]*\))*\)"
    text = re.sub(r"!\[[^\]]*\]" + target, "", text)
    text = re.sub(r"\[([^\]]*)\]" + target, r"\1", text)
    return re.sub(r"\n(?:[ \t]*\n){3,}", "\n\n", text)


def clean(body: str) -> str:
    """Clean prose markup without changing fenced code blocks or their contents."""
    parts: list[str] = []
    prose: list[str] = []
    fence: str | None = None
    for line in body.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence is None and marker:
            parts.append(_clean_markup("".join(prose)))
            prose.clear()
            fence = marker[1]
            parts.append(line)
        elif fence is not None:
            parts.append(line)
            if re.fullmatch(
                rf" {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*", line.rstrip("\r\n")
            ):
                fence = None
        else:
            prose.append(line)
    parts.append(_clean_markup("".join(prose)))
    return "".join(parts).strip()
