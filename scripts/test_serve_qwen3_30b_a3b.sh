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
REQUEST_RATE=${REQUEST_RATE:-20}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-20}
NUM_PROMPTS=${NUM_PROMPTS:-20}
INPUT_LEN=${INPUT_LEN:-2048}
OUTPUT_LEN=${OUTPUT_LEN:-1024}
# Per-request HTTP timeout. It must exceed the longest request:
# prefill + output_len x ~0.9 s/token. Default 1200s (20 min) covers
# OUTPUT_LEN=1024; override with READY_TIMEOUT when changing OUTPUT_LEN.
READY_TIMEOUT=${READY_TIMEOUT:-1200}
RESULT_FILE=${RESULT_FILE:-$PROJECT_ROOT/workspace/exp/test_serve_qwen3_30b_a3b.json}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  echo "Install dependencies with: uv sync" >&2
  exit 1
fi

mkdir -p "$(dirname -- "$RESULT_FILE")"
echo "Running test_serve benchmark against ${BASE_URL} ..."
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT/benchmarks${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m test_serve \
  --backend "$BACKEND" \
  --endpoint "$BASE_URL" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --dataset-name "$DATASET_NAME" \
  --num-prompts "$NUM_PROMPTS" \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --request-rate "$REQUEST_RATE" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --ready-timeout "$READY_TIMEOUT" \
  --save-result "$RESULT_FILE" \
  "$@"
