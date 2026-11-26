"""Paired GSM8K numeric-answer accuracy over the OpenAI Completions API."""

from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .support import ROOT, completion, read_jsonl, write_json

STRICT = re.compile(r"####\s+(-?[0-9.,]+)")
FLEXIBLE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def answer(text: str, *, strict: bool) -> str | None:
    final = text.rsplit("</think>", 1)[-1]
    matches = (STRICT if strict else FLEXIBLE).findall(final)
    if not matches:
        return None
    raw = matches[-1].replace(",", "").replace("$", "").removesuffix(".")
    try:
        return str(Decimal(raw).normalize())
    except InvalidOperation:
        return None


def evaluate(
    url,
    runtime,
    case,
    output: Path,
    *,
    limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(runtime["model"], local_files_only=True)
    raw = ROOT / "data/raw/gsm8k"
    train, test = (read_jsonl(raw / f"{split}.jsonl") for split in ("train", "test"))
    rng = random.Random(case["seed"])
    example_ids = rng.sample(range(len(train)), case["fewshot"])
    examples = [train[index] for index in example_ids]
    demonstrations = "\n\n".join(
        f"Question: {item['question']}\nAnswer: {item['answer']}" for item in examples
    )
    instructions = "Solve the math problem. End with the final numeric answer in the form #### <number>."
    question_ids = (
        list(range(len(test)))
        if limit is None
        else sorted(rng.sample(range(len(test)), min(limit, len(test))))
    )
    full_problem_count = len(question_ids)
    question_ids = question_ids[shard_index::shard_count]
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "case.json",
        {
            "configuration": case,
            "fewshot_indices": example_ids,
            "problem_indices": question_ids,
            "full_problem_count": full_problem_count,
            "shard_index": shard_index,
            "shard_count": shard_count,
        },
    )

    def evaluate_one(index):
        item = test[index]
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": instructions},
                {
                    "role": "user",
                    "content": f"{demonstrations}\n\nQuestion: {item['question']}\nAnswer:",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        gold = answer(item["answer"], strict=True)
        if gold is None:
            raise ValueError(f"GSM8K gold answer is missing a number at index {index}")
        try:
            response = completion(
                url, runtime["served_model"], prompt, **case["sampling"]
            )
        except OSError as error:
            return {"id": index, "gold": gold, "error": str(error)}
        choice = response["choices"][0]
        text = choice["text"]
        strict_answer, flexible_answer = (
            answer(text, strict=flag) for flag in (True, False)
        )
        return {
            "id": index,
            "gold": gold,
            "text": text,
            "finish": choice["finish_reason"],
            "usage": response["usage"],
            "strict_answer": strict_answer,
            "flexible_answer": flexible_answer,
            "strict_correct": strict_answer == gold,
            "flexible_correct": flexible_answer == gold,
        }

    results = []
    with (
        ThreadPoolExecutor(max_workers=case["concurrency"]) as pool,
        (output / "responses.jsonl").open("w") as stream,
    ):
        for row in pool.map(evaluate_one, question_ids):
            results.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if len(results) % 32 == 0:
                stream.flush()
                print(f"GSM8K {len(results)}/{len(question_ids)}", flush=True)
    summary = {
        "requests": len(results),
        "strict_correct": sum(row.get("strict_correct", False) for row in results),
        "flexible_correct": sum(row.get("flexible_correct", False) for row in results),
        "truncated": sum(row.get("finish") == "length" for row in results),
        "unparsed": sum(
            "error" not in row and row.get("strict_answer") is None for row in results
        ),
        "errors": sum("error" in row for row in results),
    }
    write_json(output / "summary.json", summary)
    return summary
