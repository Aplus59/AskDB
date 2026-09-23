# Knowing when not to answer

A confidently wrong number is worse than no number, because nobody knows to
check it. So the system has three outcomes rather than two: answer, ask for
clarification, or decline.

The confidence score is assembled from signals the agent already produced, so
abstention costs no extra model call. Because it depends only on recorded
values, the whole threshold sweep replays offline from a finished run in
milliseconds — the expensive part (answering 456 questions) happens once, and
the cheap part (deciding whether to stand behind each answer) replays as often
as needed.

## Which signals actually predict a wrong answer

Three were tried. Measured over 454 answered questions with a base error rate
of **0.359**:

| Signal | Fires on | Error rate when it fires | Lift |
|--------|---------:|-------------------------:|-----:|
| Unresolved self-check concern | 35 | 0.657 | **1.83×** |
| Low vocabulary coverage | 119 | 0.378 | 1.05× |
| Query needed a repair | 10 | 0.300 | **0.84×** |

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
| Coverage | 0.678 | **0.919** |
| Accuracy when answered | 0.676 | 0.666 |
| Silent error rate | 0.219 | 0.307 |
| **Abstention precision** | 0.442 | **0.676** |

The three-signal version looked better on the headline: it cut silent errors
from 35.7% to 21.9%. But it declined **a third of every question asked**, and of
those it declined only 44% would actually have been wrong — barely above the
36% base rate. It was withholding roughly eighty correct answers to suppress
sixty wrong ones.

The one-signal version declines 8% of questions and is right about two-thirds
of them. It catches fewer wrong answers in absolute terms and wastes far less.

## The full sweep

```
 clarify<  coverage   acc|ans  silent err  abst prec  answered
     0.00     0.996     0.641       0.357      1.000   454/456
     0.70     0.919     0.666       0.307      0.676   419/456
     1.01     0.000     0.000       0.000      0.362     0/456
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

One signal, firing on 7.7% of questions, with a lift of 1.83×. That is a real
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
