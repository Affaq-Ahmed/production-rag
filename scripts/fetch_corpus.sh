#!/usr/bin/env bash
# Fetch the English Kubernetes docs (concepts, tasks, tutorials, setup) into data/k8s-docs.
# Set K8S_DOCS_COMMIT to fetch an exact commit; otherwise fetches main and prints its hash.
set -euo pipefail

REPO="https://github.com/kubernetes/website"
DEST="data/k8s-docs"
SECTIONS=(
  content/en/docs/concepts
  content/en/docs/tasks
  content/en/docs/tutorials
  content/en/docs/setup
)

if [ ! -d "$DEST/.git" ]; then
  git clone --depth 1 --filter=blob:none --sparse --no-checkout "$REPO" "$DEST"
fi

cd "$DEST"
git sparse-checkout set "${SECTIONS[@]}"

if [ -n "${K8S_DOCS_COMMIT:-}" ]; then
  git fetch --depth 1 origin "$K8S_DOCS_COMMIT"
  git checkout --quiet "$K8S_DOCS_COMMIT"
else
  git checkout --quiet
fi

echo "commit: $(git rev-parse HEAD)"
echo "pages:  $(find content/en/docs -name '*.md' | wc -l | tr -d ' ')"