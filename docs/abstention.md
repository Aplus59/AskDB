# Knowing when not to answer

A confidently wrong number is worse than no number, because nobody knows to
check it. So the system has three outcomes rather than two: answer, ask for
clarification, or decline.

The confidence score is assembled from signals the agent already produced —
repairs used, whether a self-check concern survived a recheck, and how much of
the question's vocabulary appears anywhere in the schema. It costs no extra
model call, and because it depends only on recorded values, the entire
threshold sweep is replayed offline from a finished run in milliseconds.

## Sweep over 165 questions

| clarify below | coverage | accuracy when answered | silent error rate | abstention precision |
|---:|---:|---:|---:|---:|
| 0.00 | 1.000 | 0.636 | 0.364 | — |
| 0.60 | 1.000 | 0.636 | 0.364 | — |
| 0.75 | 0.994 | 0.640 | 0.358 | 1.000 |
| 0.95 | 0.945 | 0.660 | **0.321** | 0.778 |
| 1.01 | 0.000 | — | 0.000 | 0.364 |

*Silent error rate* is the share of all questions answered confidently and
wrongly. *Abstention precision* is the share of declined questions that would
have been wrong anyway.

## The signal is too coarse to be useful

Two things are true at once, and the second matters more.

**When it fires, it is usually right.** At the strictest useful threshold,
78% of the questions it declined would have been answered wrongly. That is
far better than chance, given the base error rate of 36%.

**It almost never fires.** The score is 1.0 for roughly 94% of questions,
because its three inputs are all rare: repairs happen on 3.5% of questions, an
unresolved concern on under 2%, and low vocabulary coverage on a handful. The
score is effectively binary, so there is no curve to choose an operating point
on — only two usable rows in the table above.

In recall terms: of 60 wrong answers, abstention catches **7**. The silent
error rate falls from 36.4% to 32.1%. That is a real improvement and not
nearly enough to matter.

## What would fix it

A continuous confidence signal rather than three sparse binary ones. The
standard approach is self-consistency: draw the query several times at
non-zero temperature and measure how much the *result sets* agree. Queries the
model is sure about reproduce themselves; uncertain ones scatter. Agreement is
a continuous number, so it produces an actual curve.

The cost is the obvious objection — k samples means k times the calls, which
on this free tier is the binding constraint. That trade is worth measuring
rather than assuming, in the same way the value-grounding question was.

## Honest summary

The abstention machinery is built, tested and free to sweep. What it lacks is
a signal with enough resolution to trade coverage against reliability at any
interesting operating point. Reporting it as a working feature would overstate
it; the precision figure says the idea is sound and the inputs are not yet
good enough.
