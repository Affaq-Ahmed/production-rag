import logging
from pathlib import Path

import frontmatter

from rag.ingest.models import RawDoc

logger = logging.getLogger(__name__)


def load(corpus_dir: Path, file: Path) -> RawDoc | None:
    """Load a corpus page, logging and skipping unreadable or invalid pages."""
    try:
        path = file.relative_to(corpus_dir).as_posix()
        post = frontmatter.loads(file.read_text(encoding="utf-8"))
        title = post.get("title")
        if not isinstance(title, str) or not title.strip():
            logger.warning("Skipping page without a title: %s", path)
            return None

        content_type = post.get("content_type")
        if content_type is not None and not isinstance(content_type, str):
            logger.warning("Skipping page with invalid content_type: %s", path)
            return None

        url_path = Path(path).with_suffix("")
        if url_path.name in {"_index", "index"}:
            url_path = url_path.parent
        suffix = url_path.as_posix()
        source_url = "https://kubernetes.io/docs/"
        if suffix != ".":
            source_url += suffix + "/"

        return RawDoc(
            path=path,
            source_url=source_url,
            title=title,
            content_type=content_type,
            body=post.content,
        )
    except Exception:
        logger.warning("Skipping page that could not be loaded: %s", file, exc_info=True)
        return None
