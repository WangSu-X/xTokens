#!/usr/bin/env bash
# 用 vLLM 以 TP=4 起 Qwen3.8-27B（67 的 vllm-pd-dev 容器内）。
# 用法：./vllm_run.sh [start|stop|status]，默认 start。
set -euo pipefail

MODEL=/workspace/models/Qwen3.8-27B
PORT=8001
# 系统 python3 是 vLLM 0.8.5，没有 qwen3_5，必须用这个 venv
PYTHON=/workspace/vllm-0.26.0/.venv/bin/python
PID_FILE=/root/xTokens/workerspace/vllm_run.pid
LOG_FILE=/root/xTokens/workerspace/vllm_run.log

case "${1:-start}" in
start)
    nohup env CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 HF_HUB_OFFLINE=1 \
        "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --served-model-name Qwen3.8-27B \
        --tensor-parallel-size 8 \
        --max-model-len 32768 \
        --max-num-seqs 8 \
        --gpu-memory-utilization 0.90 \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --enable-auto-tool-choice \
        --enable-prefix-caching \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > "$PID_FILE"
    echo "启动中 PID=$PID，日志 $LOG_FILE"
    # 55.5GB 权重 + torch.compile，约 5 分钟
    until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
        kill -0 "$PID" 2>/dev/null || { echo "进程退出了，日志末尾："; tail -20 "$LOG_FILE"; exit 1; }
        sleep 5
    done
    echo "就绪：http://0.0.0.0:$PORT 模型 Qwen3.8-27B"
    echo "停止：$0 stop"
    ;;
stop)
    PID=$(cat "$PID_FILE")
    kill -TERM "$PID"
    # engine 1-2 秒就停完，父进程偶尔卡在退出阶段，30 秒后强杀
    for _ in {1..30}; do
        kill -0 "$PID" 2>/dev/null || break
        sleep 1
    done
    kill -9 "$PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "已停止，端口 $PORT 释放"
    ;;
status)
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
        echo "运行中：http://127.0.0.1:$PORT"
    else
        echo "未运行"
    fi
    ;;
*)
    echo "用法：$0 [start|stop|status]" >&2
    exit 1
    ;;
esac
