# The MCP experiment tier

An on-demand testing rung for the mind's conversational output (NPC-1916, Decision 4). It sits above the simulation's behavioural suite (`run_tests.sh --behavioral-only`): it is expensive and live-only, and an orchestrator runs it when a change to the prompt, the model or the remark grammar needs measuring. It is not a per-PR gate and not a cron.

## What it measures

One recorded run answers four questions. Each has its own analysis script, so the costly run happens once and is analysed as often as wanted:

| | Question | Script |
|---|---|---|
| (a) | Does the mind name a valid, agreeing and apt remark? How many retries does that cost? | `tools/mcp_experiment_validity.py` |
| (b) | A labelled dataset for the free-text classifier (NPC-1929): words, plus the remark named | `tools/mcp_experiment_dataset.py` |
| (c) | Are replies on topic, and do personas sound different? (LLM-as-judge) | `tools/mcp_experiment_judge.py` |
| (d) | Tokens per turn, retry overhead, latency and catalogue size | `tools/mcp_experiment_cost.py` |

## What it runs

`tools/run_mcp_experiment.py` runs the real `ReflectionNode`, with its real retry budget, over every pair of a case from `tests/fixtures/mcp_experiment.py::CASES` and a persona from `PERSONAS`. Each case is an in-conversation state with known apt remarks: a greeting, a plain reply, leaving, answering "who told you?", sharing a place or a thing, correcting a stale belief, and asking who told someone. The keys come from the committed catalogue fixture, which is pinned to a game commit.

It runs in two arms:

- **`catalogue`**: the remark catalogue is present, so naming a remark is required and validated.
- **`no_catalogue`** (the red arm): the same states without the catalogue. Naming validity must collapse here, or the measured arm proves nothing. `--check-floors` fails a run whose red arm agrees as often as the measured arm.

Only the reflection node runs. Memory query and retrieval are replaced by each case's committed memories, so a run measures remark behaviour, not the vector store.

Each turn becomes one JSONL record holding:

- every attempt's prompt and raw output;
- the words, the remark named and the final parameters;
- each validation failure, the retries, and whether the turn was salvaged;
- token usage and latency;
- the case's expectations and its provenance.

`<out>.meta.json` records the mind commit, whether the tree was dirty, and the arguments. For Ollama it also records `ollama ps`, which shows the resident model's VRAM and context length.

## Backends

The backend and the model are parameters. `--backend ollama` (the default) uses `mind.apis.langchain_llm.get_local_llm`, the same `ChatOpenAI` client pointed at Ollama's OpenAI-compatible endpoint, so `LLMNode` needs no change. `--backend openrouter` uses `get_llm` and needs `OPENROUTER_API_KEY` in `credentials/api_keys.cfg`.

For a hybrid thinking model such as `qwen3:8b`, pass `--reasoning-effort none` to turn its thinking off. Do not pass it to a thinking-only model: that model then writes its reasoning into `content`, which breaks JSON parsing.

The target is a small local model. It should be fast enough that a 25th-percentile Steam-survey machine could run it, which lets the tier run often.

## Running it

From the `mind` project root (`PYTHONPATH=$PWD` because the matrix lives under `tests/fixtures`, which the wheel does not carry):

```bash
PYTHONPATH=$PWD uv run --frozen python tools/run_mcp_experiment.py --dry-run
PYTHONPATH=$PWD uv run --frozen python tools/run_mcp_experiment.py \
    --backend ollama --model qwen3:8b --reasoning-effort none --out runs/qwen3-8b.jsonl
PYTHONPATH=$PWD uv run --frozen python tools/mcp_experiment_validity.py runs/qwen3-8b.jsonl \
    --write-floors runs/floors.json       # first measured run only; never guess a floor
PYTHONPATH=$PWD uv run --frozen python tools/mcp_experiment_validity.py runs/next.jsonl \
    --check-floors runs/floors.json       # exits 1 on a violation
PYTHONPATH=$PWD uv run --frozen python tools/mcp_experiment_cost.py runs/qwen3-8b.jsonl
PYTHONPATH=$PWD uv run --frozen python tools/mcp_experiment_dataset.py runs/qwen3-8b.jsonl \
    --out runs/qwen3-8b.npc1929.jsonl
PYTHONPATH=$PWD uv run --frozen python tools/mcp_experiment_judge.py runs/qwen3-8b.jsonl \
    --backend ollama --model qwen3:8b --reasoning-effort none --out runs/qwen3-8b.judge.json
```

`--cases`, `--personas`, `--arms` and `--reps` narrow or widen a run.

The committed floors, `tests/fixtures/mcp_experiment_floors.json`, were derived from the first measured run, and its `source` field names that run. They hold only for the model they name. Re-derive them for a different model rather than reusing them.

## Reading the numbers

- **Agreement is not truthfulness.** The validator guarantees that a named remark agrees with its declaration. It cannot tell whether the words actually say that remark. The apt rate (named a key the case expects) and the judge pass are where that gets measured.
- **Retries have two causes.** The validity report splits failed attempts into `remark_failures` (the rule this tier exists for) and `other_failures` (JSON errors and other action validators). Only the first is the remark grammar's cost.
- **Effective throughput** is completion tokens divided by call wall time, prompt processing included. It is the figure a player would feel, not the model's raw decode speed.
- **A same-family judge grades its own kind.** For a number anyone acts on, judge with a different, stronger model.

The offline parts (the matrix, the record shape and all four analyses) are pinned by `tests/unit/test_mcp_experiment.py` over a hand-built fixture artifact. Nothing live runs in the test suite.
