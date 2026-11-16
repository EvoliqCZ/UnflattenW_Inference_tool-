#!/usr/bin/env bash
# Shared staleness checks for ensure-build.sh (rebuild-needed) and
# start-api.sh (recopy-needed). Split out so freshness.test.ts can source
# this file and exercise `needs_build`/`needs_copy` directly against
# disposable temp trees, instead of only re-reading the logic by eye.
set -euo pipefail

# Paths (relative to a web/ root) that affect the built dist/ output. Kept as
# a single list so ensure-build.sh's rebuild trigger and this file's own
# tests enumerate exactly the set named in the task: web/src, web/public,
# web/index.html, web/package.json, vite.config.*, web/openapi.json,
# web/codes.json, web/routes.json.
BUILD_INPUT_GLOBS=(
  "src"
  "public"
  "index.html"
  "package.json"
  "vite.config.*"
  "openapi.json"
  "codes.json"
  "routes.json"
)

# needs_build <web_dir>
# True (exit 0) when <web_dir>/dist/index.html is missing, or any tracked
# input under <web_dir> has a newer mtime than it.
#
# Why mtime and not a content hash: every build input already carries a
# meaningful mtime (edited by hand, or by `npm run generate`/`npm install`),
# so `find -newer` is a single cheap pass with no extra stamp file to keep in
# sync or itself get stale. A content hash would need to read and hash every
# source file on every invocation just to detect the common case (nothing
# changed) and would add a second piece of state (the stamp) that could drift
# from the dist/ it describes -- exactly the class of bug this fix removes.
needs_build() {
  local web_dir="$1"
  local marker="$web_dir/dist/index.html"
  if [ ! -f "$marker" ]; then
    return 0
  fi
  local pattern match
  local -a existing=()
  for pattern in "${BUILD_INPUT_GLOBS[@]}"; do
    for match in "$web_dir"/$pattern; do
      [ -e "$match" ] && existing+=("$match")
    done
  done
  if [ "${#existing[@]}" -eq 0 ]; then
    return 1
  fi
  [ -n "$(find "${existing[@]}" -newer "$marker" -type f -print -quit 2>/dev/null)" ]
}

# dist_hash <dir>
# Content hash of every file under <dir>, keyed by path relative to <dir> so
# two copies of the same dist/ hash identically regardless of where they live
# or what mtimes `cp` gave them.
dist_hash() {
  local dir="$1"
  (cd "$dir" && find . -type f -print0 | sort -z | xargs -0 sha256sum) | sha256sum | awk '{print $1}'
}

# needs_copy <src_dist_dir> <dst_dir>
# True when <dst_dir> has no index.html yet, or its content hash differs
# from <src_dist_dir>'s.
needs_copy() {
  local src="$1" dst="$2"
  if [ ! -f "$dst/index.html" ]; then
    return 0
  fi
  [ "$(dist_hash "$src")" != "$(dist_hash "$dst")" ]
}
