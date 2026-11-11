#!/usr/bin/env bash
set -euo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE=${1:?usage: round1.sh prepare|pilot|full|status}
cd "$P"; mkdir -p logs

MAIN=banking77,bitext,clinc150
HINT=hint3_sofmattress,hint3_curekart,hint3_powerplay11
BF16_REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

case "$STAGE" in
  pilot) RUN=round1_pilot; LIM="--limit 5"; LIMX="--limit 2" ;;
  full)  RUN=round1;       LIM="";          LIMX="" ;;
  prepare|status) RUN=round1 ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
A="--run-name $RUN"
log() { echo "logs/${RUN}_$1.log"; }

start() {
  local name=$1 cmd=$2
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "!! session $name already exists (attach or kill it: tmux kill-session -t $name)"; return
  fi
  tmux new-session -d -s "$name" tcsh
  tmux send-keys -t "$name" "cd $P" Enter
  tmux send-keys -t "$name" "$cmd" Enter
  echo "started $name"
}

if [[ "$STAGE" == "status" ]]; then
  tmux ls 2>/dev/null | grep bp01r1 || echo "no bp01r1 sessions"
  for f in logs/round1*_*.log; do [[ -f $f ]] && { echo "== $f"; tail -n 2 "$f"; }; done
  for r in results/round1_pilot results/round1; do
    [[ -d $r/raw ]] && { echo "== rows in $r/raw"; find "$r/raw" -name '*.jsonl' -exec wc -l {} + | sort -k2; }
  done
  exit 0
fi

if [[ "$STAGE" == "prepare" ]]; then
  start bp01r1-prep "source activate_project.csh && python -m bp01.data --hint3 |& tee -a $(log data) && python -m bp01.records --extra |& tee -a $(log records) && hf download Qwen/Qwen3.8-27B --revision $BF16_REV |& tail -3 && echo PREPARE-DONE"
  exit 0
fi

VLLM="setenv VLLM_USE_DEEP_GEMM 0 && python -m bp01.runners.vllm_llm"
EXACT="python -m bp01.runners.exact_readout"

GA=${GA:-2}; GB=${GB:-3}

start bp01r1-gpuA "source activate_clef.csh && setenv CUDA_VISIBLE_DEVICES $GA && python -m bp01.runners.clef --variant clef --tasks $HINT $A $LIM |& tee -a $(log gpuA_clef); python -m bp01.runners.clef --variant clef_flash --tasks $HINT $A $LIM |& tee -a $(log gpuA_clef_flash); source activate_project.csh && $VLLM --llm qwen --mode readout --tasks $HINT $A $LIM |& tee -a $(log gpuA_qwen_readout); $VLLM --llm qwen --mode generative --tasks $HINT $A $LIM |& tee -a $(log gpuA_qwen_gen); $EXACT --llm qwen_bf16 --tasks $MAIN --shard 0/2 $A $LIMX |& tee -a $(log gpuA_qwen_bf16_exact); echo GPUA-DONE"

start bp01r1-gpuB "source activate_pplx.csh && setenv CUDA_VISIBLE_DEVICES $GB && python -m bp01.runners.pplx --tasks $HINT $A $LIM |& tee -a $(log gpuB_pplx); source activate_project.csh && $VLLM --llm gemma --mode readout --tasks $HINT $A $LIM |& tee -a $(log gpuB_gemma_readout); $VLLM --llm gemma --mode generative --tasks $HINT $A $LIM |& tee -a $(log gpuB_gemma_gen); python -m bp01.runners.baselines --which modernbert --tasks $HINT $A $LIM |& tee -a $(log gpuB_modernbert); $VLLM --llm qwen_bf16 --mode readout --tasks $MAIN $A $LIM |& tee -a $(log gpuB_qwen_bf16_readout); $EXACT --llm qwen_bf16 --tasks $MAIN --shard 1/2 $A $LIMX |& tee -a $(log gpuB_qwen_bf16_exact); echo GPUB-DONE"

start bp01r1-cpu "source activate_project.csh && python -m bp01.runners.api --provider jev --tasks $HINT $A $LIM |& tee -a $(log cpu_jev); python -m bp01.runners.baselines --which tfidf --tasks $HINT $A $LIM |& tee -a $(log cpu_tfidf); python -m bp01.runners.baselines --which majority --tasks $HINT $A $LIM |& tee -a $(log cpu_majority); echo CPU-DONE"
start bp01r1-pplxapi "source activate_project.csh && python -m bp01.runners.api --provider pplx_api --tasks $HINT $A $LIM |& tee -a $(log cpu_pplx_api); echo PPLXAPI-DONE"

echo "All sessions started. Monitor with: bash scripts/round1.sh status"
