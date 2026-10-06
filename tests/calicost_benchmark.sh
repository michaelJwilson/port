#!/usr/bin/env bash
# CalicoST against `run_cnaster_port --sal` on the fixture docs/calicost-benchmark.md
# was measured on: dev_tree r0 (`3381575a`), drawn from
# sim/manifests/baseline/dev_tree.toml.
#
#   tests/calicost_benchmark.sh [ROOT]          # default ROOT: .cache/benchmarks/calicost_dev_tree
#
# 1. Draws the fixture into ROOT/sim and refuses it unless it hashes to 3381575a
#    (`tests.sim_stages.realization_hash`).
# 2. Runs CalicoST as #532 did: shipped `configuration_cna_multi`, `n_clones 5`,
#    uncapped. A killed attempt is rerun on the same ROOT/calicost, so CalicoST
#    resumes from its checkpoints (`tests.final_benchmark.staged`); at most
#    TRIES attempts (default 3). Its wall is the attempts' sum.
# 3. Runs `run_cnaster_port --sal`, REPEATS times on a warm numba cache (default 3),
#    the median wall.
# 4. Appends each `BENCH` line to ROOT/bench.jsonl and prints the table.
#
# CalicoST took 20,243 s here on 3 cores (#532). Run it alone on the host
# (CLAUDE.md: one measurement at a time); set LOCK to a file to hold a flock.
set -euo pipefail

ROOT=$(realpath -m "${1:-.cache/benchmarks/calicost_dev_tree}")
TRIES=${TRIES:-3}
REPEATS=${REPEATS:-3}
MANIFEST=sim/manifests/baseline/dev_tree.toml
HASH=3381575a
SAMPLE="$ROOT/sim/dev_tree/r0"
BENCH="$ROOT/bench.jsonl"

if [[ -n "${LOCK:-}" ]]; then
    exec 9>"$LOCK"
    flock 9
fi

mkdir -p "$ROOT"

if [[ ! -d "$SAMPLE" ]]; then
    python -m port.sim.draw "$MANIFEST" --into "$ROOT/sim" >&2
fi

drawn=$(python -c "from pathlib import Path; from tests.sim_stages import realization_hash; print(realization_hash(Path('$SAMPLE')))")
if [[ "$drawn" != "$HASH" ]]; then
    echo "dev_tree r0 hashes to $drawn, not $HASH: not the benchmark's fixture" >&2
    exit 1
fi

for attempt in $(seq 1 "$TRIES"); do
    line=$(python -m tests.final_benchmark "$SAMPLE" calicost --timeout 0 --n-clones 5 \
        --root "$ROOT/calicost" | grep '^BENCH ')
    echo "${line#BENCH }" | python -c "import json,sys; r=json.load(sys.stdin); r['attempt']=$attempt; print(json.dumps(r))" >>"$BENCH"
    if echo "$line" | grep -q '"finished": true'; then
        break
    fi
    echo "CalicoST attempt $attempt stopped; resuming from its checkpoints" >&2
done

python -m tests.final_benchmark "$SAMPLE" port --repeats "$REPEATS" | grep '^BENCH ' \
    | sed 's/^BENCH //' >>"$BENCH"

python - "$BENCH" <<'EOF'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
calicost = [r for r in rows if r["tool"].startswith("CalicoST")]
wall = sum(r["wall"] for r in calicost)
print("| tool | finished | clone ARI (clones) | copy ARI | exact altered | wall s | peak GB |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for r in [*calicost[-1:], *[r for r in rows if not r["tool"].startswith("CalicoST")][-1:]]:
    total = wall if r["tool"].startswith("CalicoST") else r["wall"]
    print(
        f"| {r['tool']} | {r.get('finished', True)} | {r.get('ari', '-')} ({r.get('clones', '-')}) "
        f"| {r.get('copy_ari', '-')} | {r.get('exact_altered', '-')} | {total:,.0f} | {r['peak_gb']} |"
    )
EOF
