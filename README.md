# askdb

Natural-language questions over relational databases, evaluated on
[BIRD-SQL](https://bird-bench.github.io/) Mini-Dev.

Most text-to-SQL systems have two outcomes: an answer, or a wrong answer
presented as an answer. A confidently wrong number is worse than no number,
because nobody knows to check it. This one measures how often that happens and
can decline to answer.

| | |
|---|---|
| **Execution accuracy** | **0.641** |
| Strict (multiset) comparison | 0.597 |
| Questions scored | 454 (10 of 11 databases) |
| Tokens per question | 944 |
| Cost | ~$0.48 per 1,000 questions |

Measured on BIRD Mini-Dev with `gemini-3.5-flash-lite`. The eleventh database
never ran: the free tier allows 20 requests per day per model and the quota ran
out. Every claim below links to the run that produced it, and where a result
was inconclusive it says so.

## What I built, measured, and removed

Three improvements were added on top of the basic loop. Each was measured
against it. **None of them survived**, and that is the most useful thing in
this repository.

| Change | Cost | Result | Kept |
|---|---|---|:--:|
| Example column values in the schema | +61% tokens | +2.8 points, McNemar p = 0.51 | ✗ |
| Re-checking suspicious results | ~free | fires on 7.7%, the only signal with real lift | ✓ |
| 3-sample self-consistency voting | +177% tokens | 0 fixed, 2 broken | ✗ |

Details in [docs/agent-experiments.md](docs/agent-experiments.md).

The value-grounding estimate is worth dwelling on. Measured three times on
growing samples, it read **+4.5 points, then 0.0, then +2.8** — while the code
never changed. Any of those could have been written up as the result. The first
one would have been a confident improvement claim built on a single question
moving.

What remains is the simple system: retrieve the relevant tables, draft SQL,
execute it, and repair from the database's own error message.

## Where a language model is used, and where it deliberately is not

The model writes SQL. That is all it does.

Everything else is ordinary code: schema retrieval is BM25 over table
descriptions, result comparison is set arithmetic, the confidence score is
four numbers added together, and the abstention thresholds are swept offline.
None of it calls a model, so none of it costs anything or varies between runs.

That boundary is deliberate. A system that routes every decision through a
language model is slower, more expensive and less predictable than one that
uses it for the single thing it is actually better at.

**Schema retrieval is the clearest case.** A BM25 baseline was built first
specifically so that adding embeddings could be justified. It reached
**recall@5 = 0.924** with BIRD's evidence field included, which left too little
headroom to pay for an embedding model on every query. The embeddings were
never built. See [docs/retrieval-baseline.md](docs/retrieval-baseline.md).

## How it works

```
question
   │
   ├─ schema retrieval ─────► which tables go in the prompt   (BM25, no model)
   ├─ draft ────────────────► SQL                             (model)
   ├─ execute ──────────────► rows, or the database's error
   │     └─ on error ───────► repair, bounded at 2 attempts   (model)
   ├─ self-check ───────────► does the result look wrong?     (no model)
   └─ confidence ───────────► answer, clarify, or refuse      (no model)
```

Read-only is enforced at the SQLite connection, not by prompting. A generated
`DROP TABLE` fails at execution. There is a test that sends `DELETE` through
the full HTTP path and then opens the database to confirm the rows are still
there.

## Evaluation

Queries are scored by execution: the predicted and reference queries both run
against the same database and their result rows are compared. Comparing SQL
text would be wrong — there are many correct ways to write the same query.

Three decisions shape every number reported here:

- **Row order is ignored**, matching BIRD's official set-based metric, so
  scores stay comparable to the public leaderboard.
- **A stricter multiset comparison is reported alongside it.** Set comparison
  cannot tell `COUNT` from `COUNT DISTINCT`; reporting only the official
  metric would hide that class of error.
- **Floats are rounded before comparison.** Exact float equality makes any
  query involving division randomly fail. This is a deliberate deviation from
  the official implementation.

### Failures are classified by hand, not just counted

Published analysis found annotation error rates above 50% on BIRD Mini-Dev,
with corrections moving leaderboard positions by up to nine places. A raw score
is therefore not evidence on its own.

Failures are read and labelled: genuine model error, answer-shape mismatch,
ambiguous question, benchmark annotation error, or unanswerable.

Of the first 11 failures classified, **only 2 were genuine reasoning errors**:

| Category | Count |
|---|---:|
| format_mismatch — right data, different shape | 6 |
| model_error | 2 |
| ambiguous question | 2 |
| annotation_error — the reference is wrong | 1 |

`california_schools` scores worst of all ten databases at 0.367, and this is
why: **the same concept exists under three names across three tables.** School
name is `schools.School`, `frpm."School Name"` and `satscores.sname`. The model
picks a semantically correct column that is not the one the reference happened
to pick, and set comparison marks it wrong.

Other examples: `RANK()` against `DENSE_RANK()` where the question never says
how ties break; `'well-finished'` against `'Yes'` for identical `CASE` logic;
the right month returned as `201307` instead of `07`.

Eleven of 165 failures are labelled, so treat the proportions as a direction
rather than a measurement. Two of the 500 reference queries also time out after
thirty seconds and are excluded from scoring entirely.

```bash
python scripts/audit_failures.py --results data/runs/main.jsonl --list
python scripts/audit_failures.py --results data/runs/main.jsonl --label 341 annotation_error
python scripts/audit_failures.py --results data/runs/main.jsonl
```

## Abstention

Three outcomes rather than two: answer, ask for clarification, or decline.
Everything the score uses is recorded per question, so the whole threshold
sweep replays offline in milliseconds rather than re-running the system once
per threshold.

```bash
python scripts/sweep_abstention.py --results data/runs/main.jsonl
```

### Three signals were tried; two were removed

Measured over 454 answered questions against a base error rate of 0.359:

| Signal | Fires on | Error rate when it fires | Lift |
|---|---:|---:|---:|
| Unresolved self-check concern | 35 | 0.657 | **1.83×** |
| Low vocabulary coverage | 119 | 0.378 | 1.05× |
| Query needed a repair | 10 | 0.300 | **0.84×** |

Vocabulary coverage fires on a quarter of all questions and returns the base
rate with noise. The repair penalty points the **wrong way** — a repaired query
is no likelier to be wrong, because the database rejected the first attempt and
the model corrected it under real feedback, so the surviving query has been
validated in a way the others have not.

Dropping both improved the system on every axis that matters:

| | Three signals | One signal |
|---|---:|---:|
| Coverage | 0.678 | **0.919** |
| Abstention precision | 0.442 | **0.676** |

The old configuration declined a third of all questions and was right about
those barely better than chance. What is left is precise and rare, which is a
smaller claim than a smooth curve over three, and a true one.

[docs/abstention.md](docs/abstention.md) has the full sweep.

## Running it

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev,api,data]"

.venv/bin/pytest          # no API key or dataset required
.venv/bin/ruff check .
.venv/bin/mypy src tests scripts
```

The test suite runs entirely against fixtures and fakes. The model is faked
through a narrow `ModelClient` protocol rather than mocked at the network
layer, so a fresh checkout gets a green build with no credentials.

### With a model

Put a [Google AI Studio](https://aistudio.google.com/apikey) key in `.env`:

```
GOOGLE_API_KEY=...
GOOGLE_GENAI_USE_VERTEXAI=FALSE
```

```bash
python scripts/check_model_access.py            # confirm the key works
python scripts/fetch_minidev.py                 # download the questions
python scripts/evaluate_agent.py --limit 22     # a small spread across databases
uvicorn askdb.api.app:create_app --factory      # the HTTP service
```

Evaluation runs append to disk as they go and resume where they stopped. That
is not defensive programming: the free tier allows 20 requests per day per
model, so long runs *will* be interrupted.

### In a container

```bash
docker compose up --build
```

The databases are mounted rather than copied — Mini-Dev alone is about 1.5 GB,
which has no business inside an image that is otherwise a few hundred
megabytes. The mount is read-only, so a bug in the connection layer still
cannot write to them, and the response cache lives on its own writable volume
because of that. The API key comes from the environment and is never built
into the image.

**Not verified.** Docker was not available on the machine this was developed
on, so the image has never been built. The commands it runs are the same ones
used locally, but treat this as untested until it is.

## Limitations

- **Mini-Dev schemas are small.** Nine of eleven databases have eight tables or
  fewer, so `recall@10` is close to meaningless and retrieval is not the
  bottleneck here. On Spider 2.0's enterprise schemas it would be.
- **Free-tier latency is not a measurement of this system.** The same call to
  the same model ranged from 0.6 s to 35 s within ten minutes. Latency figures
  describe Google's congestion.
- **Abstention has poor recall**, as above.
- **Sample sizes are small.** Where a comparison is inconclusive it is reported
  as inconclusive, with the significance test that says so.

## Layout

```
src/askdb/
  db/          read-only access and schema introspection
  retrieval/   BM25 over table descriptions
  llm/         caching, retries, model fallback
  agent/       tools, prompts, the loop, self-check, confidence
  eval/        execution scoring, runs, failure audit, abstention sweep
  api/         the HTTP service
scripts/       dataset fetch, evaluation, audit, sweeps
docs/          measured results
```
