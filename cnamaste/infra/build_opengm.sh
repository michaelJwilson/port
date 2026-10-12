#!/usr/bin/env bash
# Build cnamaste's OpenGM library, `libcnamaste_opengm.so`, into a cache (S2, T- #836).
#
# Copied from snakes_and_ladders' infra/build_opengm.sh (MIT, M. J. Wilson; sal #1279), names changed.
#
# OpenGM is not on PyPI (#962), and its inference is header-only C++
# templates. This fetches the pinned commit's headers, compiles
# `infra/opengm/cnamaste_opengm.cxx` against them with the host's g++, and writes
# the library, the commit and the licence terms into one directory, which
# `cnamaste.external` finds and its tests skip without. No Boost, no
# HDF5, no Python binding and no downloaded external solver is built: every
# algorithm in scope is OpenGM's own MIT code (#1279's spike), and the
# min s-t cut alpha expansion and the swap need is the file's own.
#
# Optionally, CNAMASTE_OPENGM_MAXFLOW=kolmogorov also fetches Kolmogorov's
# maxflow v3.03 (github.com/opengm/maxflow, pinned; GPL-3.0) and compiles it in
# through OpenGM's own wrapper, adding `expansion_kolmogorov` and
# `swap_kolmogorov`. The library is then GPL-3.0; `BACKENDS` lists the cuts built.
#
#   infra/build_opengm.sh            # build, or report the cached build current
#   CNAMASTE_OPENGM_MAXFLOW=kolmogorov infra/build_opengm.sh
#   CNAMASTE_OPENGM_HOME=/some/dir infra/build_opengm.sh
#
# Idempotent: a build whose stamp (commit, driver source, flags) matches is
# kept, and anything else is rebuilt. The directory is
# `$CNAMASTE_OPENGM_HOME`, else `${XDG_CACHE_HOME:-$HOME/.cache}/cnamaste/opengm`, the
# rule `cnamaste.external.home` reads.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

#: The commit built: OpenGM's last, 2019-07-26; the project is unmaintained.
commit="decdacf4caad223b0ab5478d38a855f8767a394f"
repository="https://github.com/opengm/opengm"
home="${CNAMASTE_OPENGM_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/cnamaste/opengm}"
source_file="$root/infra/opengm/cnamaste_opengm.cxx"
flags="-std=c++11 -O2 -shared -fPIC -w"
compiler="${CXX:-g++}"
maxflow="${CNAMASTE_OPENGM_MAXFLOW:-dinic}"
#: Kolmogorov's maxflow v3.03 as OpenGM packages it, 2014-11-26.
maxflow_commit="998bbe645d198bbe0da0c2dc9306901a31d4ebd5"
maxflow_repository="https://github.com/opengm/maxflow"
case "$maxflow" in
  dinic) ;;
  kolmogorov) flags="$flags -DCNAMASTE_WITH_KOLMOGOROV" ;;
  *) echo "CNAMASTE_OPENGM_MAXFLOW is dinic or kolmogorov, not $maxflow" >&2; exit 2 ;;
esac

stamp="$commit $maxflow $(sha256sum "$source_file" | cut -d' ' -f1) $compiler $flags"
if [ -f "$home/STAMP" ] && [ "$(cat "$home/STAMP")" = "$stamp" ] && [ -f "$home/libcnamaste_opengm.so" ]; then
  echo "opengm: current at $home"
  exit 0
fi

mkdir -p "$home"
# fetch <directory> <repository> <commit>: a shallow checkout of one commit, kept while it matches.
fetch() {
  if [ "$(git -C "$1" rev-parse HEAD 2>/dev/null)" != "$3" ]; then
    rm -rf "$1"
    git init -q "$1"
    git -C "$1" fetch -q --depth 1 "$2" "$3"
    git -C "$1" checkout -q FETCH_HEAD
  fi
}
fetch "$home/src" "$repository" "$commit"
extra=()
if [ "$maxflow" = kolmogorov ]; then
  fetch "$home/maxflow" "$maxflow_repository" "$maxflow_commit"
  # NB OpenGM's wrapper includes <maxflowlib.h>, the name its CMake gave the patched copy.
  mkdir -p "$home/shim"
  echo '#include <maxflow/graph.h>' > "$home/shim/maxflowlib.h"
  extra=(-I"$home/shim" -I"$home/maxflow/include" "$home/maxflow/src/graph.cpp" "$home/maxflow/src/maxflow.cpp")
fi

start=$(date +%s)
# shellcheck disable=SC2086 # the flags are words
"$compiler" $flags -I"$home/src/include" "${extra[@]}" "$source_file" -o "$home/libcnamaste_opengm.so.tmp"
mv "$home/libcnamaste_opengm.so.tmp" "$home/libcnamaste_opengm.so"
echo "$commit" > "$home/COMMIT"
cat > "$home/LICENCES" <<TERMS
OpenGM $commit ($repository): MIT, Licence-OpenGM.txt.
Built: OpenGM's header-only ICM, loopy BP, A*, TRWSi (Savchynskyy's TRW-S),
dual decomposition (subgradient, dynamic-programming subproblems), alpha
expansion and alpha-beta swap; the min s-t cut is infra/opengm/cnamaste_opengm.cxx's.
Not fetched or built: the externals OpenGM's CMake downloads (Kolmogorov's
TRW-S v1.3 and MRF-LIB, QPBO v1.3: research-only or GPL; gco-v3.0:
research-only; IBFS, AD3, ConicBundle, and the rest).
TERMS
if [ "$maxflow" = kolmogorov ]; then
  cat >> "$home/LICENCES" <<TERMS
Kolmogorov's maxflow v3.03 $maxflow_commit ($maxflow_repository): GPL-3.0,
compiled in: this library is GPL-3.0.
TERMS
  echo "dinic kolmogorov" > "$home/BACKENDS"
else
  echo "Not built: Kolmogorov's maxflow (CNAMASTE_OPENGM_MAXFLOW=kolmogorov)." >> "$home/LICENCES"
  echo "dinic" > "$home/BACKENDS"
fi
echo "$stamp" > "$home/STAMP"
echo "opengm: built $home/libcnamaste_opengm.so in $(( $(date +%s) - start )) s"
