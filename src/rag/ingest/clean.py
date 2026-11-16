"""Convert Hugo markup to searchable text while preserving fenced code."""

import re

# Hugo maps {{% heading "key" %}} to fixed English strings. Unknown keys fall back to the key.
HEADINGS = {
    "whatsnext": "What's next",
    "prerequisites": "Before you begin",
    "objectives": "Objectives",
    "cleanup": "Clean up",
    "synopsis": "Synopsis",
    "options": "Options",
    "seealso": "See also",
}

# "What's next" sections are navigation: lists of related page titles whose links are already
# stripped. As chunks they match related queries without containing answers (distractors), so
# they are dropped like other page navigation. See NOTES.md, "Dropping What's next sections".
# Matches the shortcode form and the plain-Markdown heading a few pages write by hand.
_DROPPED_SECTION = re.compile(
    r"^ {0,3}(#{1,6})[ \t]+"
    r"(?:{{%\s*heading\s+([\"'])whatsnext\2\s*%}}|what's next)[ \t]*$",
    re.IGNORECASE,
)
_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _closes(fence: str, line: str) -> bool:
    return (
        re.fullmatch(rf" {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*", line.rstrip("\r\n"))
        is not None
    )


def _drop_navigation(body: str) -> str:
    """Remove each "What's next" section up to the next heading at its level or higher."""
    kept: list[str] = []
    fence: str | None = None
    dropping: int | None = None
    for line in body.splitlines(keepends=True):
        if fence is not None:
            if _closes(fence, line):
                fence = None
        elif dropped := _DROPPED_SECTION.match(line.rstrip("\r\n")):
            dropping = len(dropped[1])
            continue
        else:
            heading = _HEADING.match(line)
            if dropping is not None and heading and len(heading[1]) <= dropping:
                dropping = None
            if marker := _FENCE.match(line):
                fence = marker[1]
        if dropping is None:
            kept.append(line)
    return "".join(kept)


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
    # The page already writes the "##"; the shortcode supplies only the heading text.
    text = re.sub(
        r"""{{%\s*heading\s+(["'])(.*?)\1\s*%}}""",
        lambda m: HEADINGS.get(m[2], m[2]),
        text,
    )
    # Hugo heading IDs ("## Pod phase {#pod-phase}") would otherwise leak into heading paths.
    # Only heading lines: fenced code never reaches this function, and prose may mention {#id}.
    text = re.sub(r"^(#{1,6} .*?)[ \t]*\{#[^}\n]*\}[ \t]*$", r"\1", text, flags=re.MULTILINE)
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
    for line in _drop_navigation(body).splitlines(keepends=True):
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
