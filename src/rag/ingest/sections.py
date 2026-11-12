"""Split Markdown into sections with hierarchical heading paths."""

import re

from rag.ingest.models import Section


def split_sections(title: str, markdown: str) -> list[Section]:
    sections: list[Section] = []
    headings: list[tuple[int, str]] = []
    lines: list[str] = []
    fence: str | None = None

    def flush() -> None:
        text = "".join(lines).strip()
        if text:
            sections.append(Section((title, *(name for _, name in headings)), text))
        lines.clear()

    for line in markdown.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence is not None:
            lines.append(line)
            if re.fullmatch(
                rf" {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*",
                line.rstrip("\r\n"),
            ):
                fence = None
            continue
        if marker:
            fence = marker[1]
            lines.append(line)
            continue

        heading = re.match(r"^(#{1,4}) (.*)$", line.rstrip("\r\n"))
        if heading:
            flush()
            level = len(heading[1])
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, heading[2].strip()))
        else:
            lines.append(line)

    flush()
    return sections
