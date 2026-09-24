#!/bin/bash
# usage: run_sim.sh <easy|hard> <arm> [pure] [oracle]
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH=$PWD/python:$PWD
case "$2" in base) f="";; mix) f="--clone-mixture";; cap20) f="--clone-mixture --mixture-cap 0.2";; anneal) f="--clone-mixture --mixture-anneal 0.1,0.5";; esac
v=""; tag="sim_$1"
[ "$3" = pure ] && { v="$v --pure"; tag="${tag}_pure"; }
[ "$3" = admixed ] && tag="${tag}_admixed"
[ -n "$NF" ] && { v="$v --pure --normal-fraction $NF"; tag="${tag}_nf${NF//,/-}"; }
[ "$4" = oracle ] && { v="$v --oracle-start"; tag="${tag}_oracle"; }
tag="$tag-$2${MIX_TAG:+-$MIX_TAG}"
timeout 3600 .venv/bin/python -u -W ignore python/port/sandbox/admixture/probes/sim_probe.py --sample $1 $v -- --sal $f > ${OUT:-/tmp}/$tag.log 2>&1
grep -E "^SIM |^MIXFIT|^MIXCONF|^PLANTED" ${OUT:-/tmp}/$tag.log | sed "s/^/$tag /" > ${OUT:-/tmp}/$tag.out
