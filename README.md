# askdb

Natural-language questions over relational databases, evaluated on
[BIRD-SQL](https://bird-bench.github.io/) Mini-Dev.

Most text-to-SQL systems have two outcomes: an answer, or a wrong answer
presented as an answer. A confidently wrong number is worse than no number,
because nobody knows to check it. This one measures how often that happens and
can decline to answer.

> **Headline numbers are filled in from the full 500-question run.** Everything
> below this line is measured, and every claim links to the run that produced
> it. Where a result was inconclusive, it says so.

## What I built, measured, and removed

Three improvements were added on top of the basic loop. Each was measured
against it. **None of them survived**, and that is the most useful thing in
this repository.

| Change | Cost | Result | Kept |
|---|---|---|:--:|
| Example column values in the schema | +61% tokens | +2.8 points, McNemar p = 0.51 | ✗ |
| Re-checking suspicious results | ~free | fires on 2% of questions | ✓ |
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
ambiguous question, benchmark annotation error, or unanswerable. In the first
sample of five, one was a benchmark bug — the reference query tested the same
column twice — and two returned the right data in a different shape.

```bash
python scripts/audit_failures.py --results data/runs/main.jsonl --list
python scripts/audit_failures.py --results data/runs/main.jsonl --label 341 annotation_error
python scripts/audit_failures.py --results data/runs/main.jsonl
```

## Abstention

Three outcomes rather than two. The confidence score combines repairs used,
whether a self-check concern survived, and how much of the question's
vocabulary appears in the schema — all recorded per question, so the whole
threshold sweep replays offline in milliseconds rather than re-running the
system once per threshold.

```bash
python scripts/sweep_abstention.py --results data/runs/main.jsonl
```

**Honest status:** when it fires it is usually right (78% of declined questions
would have been wrong), but it fires on about 5% of questions, so it catches
roughly one wrong answer in nine. The signal has good precision and poor
recall. [docs/abstention.md](docs/abstention.md) sets out what would fix it.

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
