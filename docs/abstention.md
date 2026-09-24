# Knowing when not to answer

A confidently wrong number is worse than no number, because nobody knows to
check it. So the system has three outcomes rather than two: answer, ask for
clarification, or decline.

The confidence score is assembled from signals the agent already produced, so
abstention costs no extra model call. Because it depends only on recorded
values, the whole threshold sweep replays offline from a finished run in
milliseconds — the expensive part (answering 500 questions) happens once, and
the cheap part (deciding whether to stand behind each answer) replays as often
as needed.

## Which signals actually predict a wrong answer

Three were tried. Measured over 498 scorable questions with a base error rate
of **0.367**:

| Signal | Fires on | Error rate when it fires | Lift |
|--------|---------:|-------------------------:|-----:|
| Unresolved self-check concern | 36 | 0.667 | **1.81×** |
| Low vocabulary coverage | 126 | 0.397 | 1.08× |
| Query needed a repair | 10 | 0.300 | **0.82×** |

**Only the first predicts anything.**

*Vocabulary coverage* — how much of the question's wording appears anywhere in
the schema — fires on more than a quarter of all questions and returns the base
rate with noise on top. It sounded like it should work. It does not.

*The repair penalty points the wrong way.* A query that needed repairing is, if
anything, slightly **less** likely to be wrong. That is obvious in hindsight: a
repair means the database rejected the first attempt and the model corrected it
against a real error message, so the surviving query has been validated in a
way a never-repaired one has not. Penalising it was penalising evidence of
correctness.

Both were removed.

## What that did to the sweep

| | Three signals | One signal |
|---|---:|---:|
| Coverage | 0.684 | **0.924** |
| Accuracy when answered | 0.667 | 0.656 |
| Silent error rate | 0.228 | 0.318 |
| **Abstention precision** | 0.449 | **0.684** |

Both columns are recomputed from the same finished run with the same
accounting, the three-signal one at the threshold where any single signal is
enough to decline — which is how it was originally configured.

The three-signal version looked better on the headline: it cut silent errors
from 36.6% to 22.8%. But it declined **a third of every question asked**, and of
those it declined only 45% would actually have been wrong — barely above the
37% base rate. It was withholding roughly eighty-seven correct answers to
suppress seventy-one wrong ones.

The one-signal version declines 8% of questions and is right about two-thirds
of them. It catches fewer wrong answers in absolute terms and wastes far less.

## The full sweep

```
 clarify<  coverage   acc|ans  silent err  abst prec  answered
     0.00     0.996     0.633       0.366      1.000   498/500
     0.70     0.924     0.656       0.318      0.684   462/500
     1.01     0.000     0.000       0.000      0.370     0/500
```

*Silent error rate* is the share of all questions answered confidently and
wrongly. *Abstention precision* is the share of declined questions that would
have been wrong anyway.

## A threshold bug the reduction exposed

With only the concern signal, a flagged answer scores exactly `1.0 - 0.4 = 0.6`
— and the default `clarify_below` was also `0.6`. The comparison is strict, so
`0.6 < 0.6` is false and the one signal that works would never have fired at
the default setting. The threshold now sits at `0.7`.

This only became visible after the other two signals were removed. While they
were still in play, something always dragged the score below 0.6 and the
boundary was never tested.

## Honest status

One signal, firing on 7.2% of questions, with a lift of 1.81×. That is a real
effect and a small one: it removes about a seventh of the silent errors.

Vocabulary coverage is still computed and stored on every result, just not
scored. Re-testing it against a larger run costs nothing and re-answers
nothing.

What would improve recall is a signal that varies per question rather than
firing rarely. Self-consistency was tried for exactly that reason and is
written up in [agent-experiments.md](agent-experiments.md): agreement across
repeated draws does correlate with correctness (0.727 / 0.500 / 0.000 across
unanimous, majority and split), but with three draws it takes only three
values, 85% of questions are unanimous, and the sampling cost nearly tripled
the token bill while breaking two questions and fixing none.
