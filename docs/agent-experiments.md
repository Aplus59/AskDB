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

Paired comparison over the 108 questions both configurations answered:

| | Baseline | Grounding + self-check |
|---|---:|---:|
| Correct | 73 / 108 | 76 / 108 |
| Execution accuracy | 0.676 | 0.704 |
| Tokens per question | 843 | **1357** (+61%) |

The changes fixed 6 questions and broke 3, for a net gain of three.

## The gain is not distinguishable from chance

Nine questions changed verdict. Under the null hypothesis that the changes do
nothing, each of those nine is a coin flip, so McNemar's exact test applies:

    6 fixed, 3 broken, two-sided p = 0.51

That is as close to "no evidence" as a result gets. With three regressions,
twelve fixes would be needed to reach p < 0.05.

So the honest summary is: **+2.8 points that cannot be distinguished from
noise, against +61% tokens that is certain.** One side of that trade is
measured and the other is not.

Measured three times now, on progressively larger samples:

| Sample | Apparent gain | Verdict |
|---|---|---|
| n=22 | +4.5 points | One question moving |
| n=57 paired | 0.0 points | Nothing |
| n=108 paired | +2.8 points | p = 0.51 |

The estimate has swung from +4.5 to 0.0 to +2.8 as the sample grew. That
instability is itself the finding: at these sample sizes the measurement is
dominated by which questions happened to be drawn.

The self-check fired on 1 question in 57. At that rate it cannot move the
score in either direction, and these runs say nothing about whether its
judgement is good — only that it is rare.

The full 500-question run since settled that: it fires on 7.2% of questions,
and when it fires the answer is wrong 66.7% of the time against a base rate of
36.7%. See [abstention.md](abstention.md).

## Decision

**Value grounding is off by default.** The cost is certain and the benefit is
not demonstrated. The flag remains so the experiment can be re-run on all 500
questions, where an effect of this size would become measurable — roughly five
times the current sample is needed to resolve a 3-point difference. That re-run
has not been done: the 500-question run reported in the README is the default
configuration, with grounding off.

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
python scripts/evaluate_agent.py --limit 110 --value-grounding \
    --output data/runs/full_110.jsonl
python scripts/evaluate_agent.py --limit 110 --no-self-check \
    --output data/runs/base_110.jsonl
```

Value grounding is off by default and the self-check is on, so the treatment
run opts into grounding and the baseline run opts out of the self-check.

Runs resume: re-running with the same output file skips questions already
answered.
