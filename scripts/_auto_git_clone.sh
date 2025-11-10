# Some dependencies in pyproject.toml needs to be retreived from git repository, not from pypi.
# Those are specified in [tool.uv.sources] with "git = & rev =".
# This script clones them to /external/ directory to check implementation of those.
# So this script is not necessary.

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p external

uv run python - <<'PY' | while IFS=$'\t' read -r name url rev; do
  dest="external/$name"
  if [[ -d "$dest/.git" ]]; then
    echo "skip: $name (already cloned)"
    continue
  fi
  git clone "$url" "$dest"
  git -C "$dest" checkout --detach "$rev"
done
import sys
try:
    import tomllib as tomli  # Py3.11+
except Exception:
    import tomli           # fallback

with open("pyproject.toml", "rb") as f:
    d = tomli.load(f)

src = (d.get("tool") or {}).get("uv", {}).get("sources", {}) or {}

def entries(v):
    if isinstance(v, dict):
        yield v
    elif isinstance(v, list):
        for i in v:
            if isinstance(i, dict):
                yield i

errs = []
for name, v in src.items():
    for it in entries(v):
        git = it.get("git")
        if not git:
            continue
        rev = it.get("rev")
        if not rev:
            errs.append((name, git))
        else:
            print(f"{name}\t{git}\t{rev}")

if errs:
    for name, git in errs:
        print(f"ERROR: [tool.uv.sources].{name} は git を指定していますが rev がありません (repo: {git})", file=sys.stderr)
    sys.exit(1)
PY
