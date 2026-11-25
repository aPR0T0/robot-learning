# Sourced by setup_*.sh: check out each upstream repo at the commit this repo pins for it (thirdparty/<name>).
#   REV=$(pinned steel)                       # the submodule's commit; empty outside a git checkout
#   clone_at DIR URL REV [--depth 1]          # clone if missing, then check out REV (fetched if not local)
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

pinned() {
  git -C "$REPO" ls-files -s "thirdparty/$1" 2>/dev/null | awk '$1 == "160000" {print $2}'
}

clone_at() {
  local dir=$1 url=$2 rev=${3:-} depth=()
  [ "${4:-}" = --depth ] && depth=(--depth "$5")
  [ -d "$dir/.git" ] || git clone -q "${depth[@]}" "$url" "$dir"
  if [ -z "$rev" ]; then
    echo "    $(basename "$dir"): no pin (not a git checkout of robot-learning); keeping $(git -C "$dir" rev-parse --short HEAD)"
    return
  fi
  if ! git -C "$dir" checkout -q "$rev" 2>/dev/null; then
    git -C "$dir" fetch -q "${depth[@]}" origin "$rev" || { echo "!! $url has no $rev" >&2; return 1; }
    git -C "$dir" checkout -q FETCH_HEAD || return 1
  fi
  echo "    $(basename "$dir") at $(git -C "$dir" rev-parse --short HEAD)"
}
