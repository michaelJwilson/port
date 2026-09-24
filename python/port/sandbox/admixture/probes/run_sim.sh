#!/bin/bash
# One simulated sample through tests.sim_audit with --sal, printing the
# mixture's fits (#380 design study). Writes $OUT/<tag>.log and .out.
#
#   run_sim.sh <easy|hard> <base|mix> [pure|admixed] [oracle]
#
# NF=f1,f2,f3        plant these normal fractions per tumour clone
# WINDOW=x0,x1,y0,y1 crop to a contiguous window of spots first
# PORT_SIM_CACHE=dir reuse generated samples across runs
# MIX_VARIANTS=1     run the sandbox variants (MIX_ROW_MODE, MIX_ENTROPY, MIX_STARTS)
cd "$(git rev-parse --show-toplevel)" || exit 1
export PYTHONPATH=$PWD/python:$PWD

flags=""; [ "$2" = mix ] && flags="--clone-mixture"
sample=""; tag="sim_$1"
[ "$3" = pure ] && { sample="$sample --pure"; tag="${tag}_pure"; }
[ "$3" = admixed ] && tag="${tag}_admixed"
[ -n "$NF" ] && { sample="$sample --pure --normal-fraction $NF"; tag="${tag}_nf${NF//,/-}"; }
[ -n "$WINDOW" ] && { sample="$sample --window $WINDOW"; tag="${tag}_w${WINDOW//,/-}"; }
[ "$4" = oracle ] && { sample="$sample --oracle-start"; tag="${tag}_oracle"; }
tag="$tag-$2${MIX_TAG:+-$MIX_TAG}"
log="${OUT:-/tmp}/$tag.log"

timeout 3600 .venv/bin/python -u -W ignore python/port/sandbox/admixture/probes/sim_probe.py \
  --sample "$1" $sample -- --sal $flags > "$log" 2>&1
grep -E "^(SIM|MIXFIT|MIXCONF|PLANTED|DECODE|WINDOW) " "$log" | sed "s/^/$tag /" > "${log%.log}.out"
