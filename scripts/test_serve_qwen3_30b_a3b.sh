#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

HOST=${XTOKENS_HOST:-127.0.0.1}
PORT=${XTOKENS_PORT:-8991}
MODEL=${XTOKENS_SERVED_MODEL_NAME:-qwen3-30b-a3b}
# Tokenizer used to generate prompts with an exact token count; keep it in
# sync with the served model so --input-len/--output-len are measured in the
# same units the server uses (a fallback word-splitter would over-count).
TOKENIZER=${TOKENIZER:-${XTOKENS_HF_MODEL:-/model}}
BASE_URL="http://${HOST}:${PORT}"
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
BACKEND=${BACKEND:-openai}
DATASET_NAME=${DATASET_NAME:-random}
SAMPLE_POOL_SIZE=${SAMPLE_POOL_SIZE:-8}
NUM_PROMPTS=${NUM_PROMPTS:-16}
OUTPUT_LEN=${OUTPUT_LEN:-256}
# Space-separated matrix values. Override these environment variables to run
# a different matrix without changing the script.
INPUT_LENS=${INPUT_LENS:-"1024 4096 8192"}
CONCURRENCIES=${CONCURRENCIES:-"8 16 32"}
REQUEST_RATE=${REQUEST_RATE:-inf}
# Per-request HTTP timeout. It must exceed the longest request:
# prefill + output_len x ~0.9 s/token. Default 1200s covers the longest
# default matrix case; override with READY_TIMEOUT when changing lengths.
READY_TIMEOUT=${READY_TIMEOUT:-1200}
RESULT_DIR=${RESULT_DIR:-$PROJECT_ROOT/workspace/exp/test_serve_qwen3_30b_a3b_matrix}

read -r -a INPUT_LENGTHS <<< "$INPUT_LENS"
read -r -a CONCURRENCY_VALUES <<< "$CONCURRENCIES"

if ((${#INPUT_LENGTHS[@]} == 0 || ${#CONCURRENCY_VALUES[@]} == 0)); then
  echo "INPUT_LENS and CONCURRENCIES must each contain at least one value" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  echo "Install dependencies with: uv sync" >&2
  exit 1
fi

mkdir -p "$RESULT_DIR"
echo "Running test_serve benchmark matrix against ${BASE_URL} ..."
cd "$PROJECT_ROOT"

for INPUT_LEN in "${INPUT_LENGTHS[@]}"; do
  for MAX_CONCURRENCY in "${CONCURRENCY_VALUES[@]}"; do
    RESULT_FILE="$RESULT_DIR/i${INPUT_LEN}_o${OUTPUT_LEN}_c${MAX_CONCURRENCY}.json"
    echo "Running: input=${INPUT_LEN} output=${OUTPUT_LEN} " \
      "pool=${SAMPLE_POOL_SIZE} requests=${NUM_PROMPTS} " \
      "concurrency=${MAX_CONCURRENCY}"
    PYTHONPATH="$PROJECT_ROOT/benchmarks${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON_BIN" -m test_serve \
      --backend "$BACKEND" \
      --endpoint "$BASE_URL" \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --dataset-name "$DATASET_NAME" \
      --sample-pool-size "$SAMPLE_POOL_SIZE" \
      --num-prompts "$NUM_PROMPTS" \
      --input-len "$INPUT_LEN" \
      --output-len "$OUTPUT_LEN" \
      --request-rate "$REQUEST_RATE" \
      --max-concurrency "$MAX_CONCURRENCY" \
      --ready-timeout "$READY_TIMEOUT" \
      --save-result "$RESULT_FILE" \
      "$@"
  done
done
