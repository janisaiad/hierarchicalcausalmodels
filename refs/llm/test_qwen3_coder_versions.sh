#!/usr/bin/env bash
# test all Qwen3 Coder variants on 2x L4 (48GB) - run on server in ~/llm
set -e
VENV=/root/llm/.venv
PORT=8000
LOG_DIR=/root/llm/test_logs
RESULTS=/root/llm/test_results.txt
mkdir -p "$LOG_DIR"

stop_vllm() {
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  fuser -k ${PORT}/tcp 2>/dev/null || true
  sleep 5
  while curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do sleep 2; done
  sleep 2
}

run_one() {
  local model="$1"
  shift
  local log="$LOG_DIR/$(echo "$model" | tr '/' '_').log"
  echo "=== Testing $model ===" >> "$RESULTS"
  stop_vllm
  $VENV/bin/python -m vllm.entrypoints.openai.api_server \
    "$model" \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port $PORT \
    --trust-remote-code \
    "$@" \
    >> "$log" 2>&1 &
  pid=$!
  max_wait=300
  for i in $(seq 1 $max_wait); do
    if curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then break; fi
    sleep 2
  done
  if ! curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    kill $pid 2>/dev/null || true
    stop_vllm
    echo "FAIL (timeout or no /v1/models)" >> "$RESULTS"
    return 0
  fi
  model_id=$(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else '')" 2>/dev/null || echo "")
  chat_ok=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":5}" | python3 -c "import sys,json; r=json.load(sys.stdin); print('ok' if 'choices' in r and r.get('choices') else 'err')" 2>/dev/null || echo "err")
  kill $pid 2>/dev/null || true
  stop_vllm
  if [ "$chat_ok" = "ok" ]; then
    echo "OK (model_id=$model_id)" >> "$RESULTS"
  else
    echo "FAIL (chat: $chat_ok)" >> "$RESULTS"
  fi
}

echo "Qwen3 Coder variant test on $(hostname) $(date)" > "$RESULTS"

# 0.6B: no overrides
run_one "Qwen/Qwen3-0.6B" || true

# clear HF cache for 30B so config is re-fetched (avoid stale 0.6B resolution)
rm -rf /root/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct 2>/dev/null || true
rm -rf /root/.cache/huggingface/hub/models--QuantTrio--Qwen3-Coder-30B-A3B-Instruct-AWQ 2>/dev/null || true
# 30B MoE: force architecture
run_one "Qwen/Qwen3-Coder-30B-A3B-Instruct" --hf-overrides '{"architectures": ["Qwen3MoeForCausalLM"]}' --max-model-len 8192 || true
run_one "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8" --hf-overrides '{"architectures": ["Qwen3MoeForCausalLM"]}' --max-model-len 8192 || true
run_one "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ" --hf-overrides '{"architectures": ["Qwen3MoeForCausalLM"]}' --max-model-len 8192 || true

# Coder-Next 80B AWQ: no quantization flag (repo is pre-AWQ)
run_one "cyankiwi/Qwen3-Coder-Next-AWQ-4bit" --max-model-len 32768 || true

echo "--- Done. Results in $RESULTS ---"
cat "$RESULTS"
