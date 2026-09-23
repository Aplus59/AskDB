# Does showing the model more actually help?

Two changes were added to the agent after the first failure audit, both
motivated by measured failures rather than intuition:

- **Value grounding** — example values for text columns inline in the schema,
  added because a query filtered `A3 = 'East Bohemia'` against a column
  storing `east Bohemia` and returned zero rows with no error.
- **Self-check** — re-examining a query that executed cleanly but returned a
  result that looks wrong (no rows, a lone NULL, many rows for a "how many"
  question), added because the repair loop only sees queries the database
  rejects, and the database rejects almost nothing.

## Result

Paired comparison over the 57 questions both configurations answered:

| | Baseline | Grounding + self-check |
|---|---:|---:|
| Correct | 35 / 57 | 35 / 57 |
| Execution accuracy | 0.614 | 0.614 |
| Tokens per question | 943 | **1667** |

**Identical accuracy at 77% more tokens.** Two questions were fixed by the
changes and two were broken by them.

An earlier n=22 sample had shown grounding at +4.5 points. That was one
question moving, and it did not survive a larger sample. The number was noise,
and reporting it as an improvement would have been wrong.

## What this can and cannot establish

n=57 with 22 failures. A two-question swing is 3.5 points, so this rules out a
large effect, not a small one. The honest claim is: *no effect larger than
roughly ±5 points, at a certain cost of +77% tokens.*

The self-check fired on 1 of 57 questions. At that rate it cannot move the
score much in either direction, and this run says nothing useful about whether
its judgement is good — only that it is rare.

## Decision

**Value grounding is off by default.** It has a certain cost and no
demonstrated benefit. The flag remains so the experiment can be re-run on the
full 500 questions, where a small effect would become visible.

**The self-check stays on.** Its cost when it does not fire is a few
microseconds of Python, and it only spends a model call on results that are
almost certainly wrong. Cheap insurance, still unproven.

## Why the baseline run was short

The baseline run answered 57 of 110. The rest were lost to
`[Errno 11001] getaddrinfo failed` — a DNS failure that was not classified as
retryable, so those questions errored out instead of backing off and trying
again. Connection and timeout errors are now treated as transient, with tests
covering both.

That is the more useful finding of the two: the retry policy was written for
API errors and had never been tested against the network simply not being
there.

## Reproducing

```bash
python scripts/evaluate_agent.py --limit 110 --output data/runs/full_110.jsonl
python scripts/evaluate_agent.py --limit 110 --no-value-grounding --no-self-check \
    --output data/runs/base_110.jsonl
```

Runs resume: re-running with the same output file skips questions already
answered.
