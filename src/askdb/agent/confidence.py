"""Deciding whether to answer at all.

Most text-to-SQL systems have two outcomes: an answer, or a wrong answer
presented as an answer. A confidently wrong number is worse than no number,
because nobody knows to check it. This adds a third outcome — ask, or decline.

The score is assembled from signals the agent has already produced, so
abstention costs no extra model call. That matters: a system that spends a
call deciding whether to spend a call has doubled its cost before answering
anything.

Three signals were tried. Measured over 454 answered questions against a base
error rate of 0.359:

    unresolved concern   fires on  35   error rate 0.657   lift 1.83x
    low vocab coverage   fires on 119   error rate 0.378   lift 1.05x
    needed a repair      fires on  10   error rate 0.300   lift 0.84x

Only the first predicts anything. Vocabulary coverage fires on a quarter of
all questions and returns the base rate with noise. The repair penalty points
the wrong way, which is obvious in hindsight: a repair means the database
rejected the query and the model corrected it under real feedback, so the
surviving query has been validated in a way the others have not.

Both were removed. What is left is one signal that is precise and rare, which
is a smaller claim than a smooth curve over three, and a true one.
"""

from dataclasses import dataclass

from askdb.retrieval.documents import tokenize

ANSWER = "answer"
CLARIFY = "clarify"
REFUSE = "refuse"

# An unresolved concern scores exactly 1.0 - 0.4 = 0.6, so the threshold
# has to sit above it. At 0.6 the comparison is strict and the only signal
# that predicts anything would never fire.
DEFAULT_CLARIFY_BELOW = 0.7
DEFAULT_REFUSE_BELOW = 0.3

UNRESOLVED_CONCERN_PENALTY = 0.4

# Disagreement between repeated draws is the only continuous input here.
# The others fire on a few percent of questions, which left the score
# effectively binary and gave the thresholds nothing to sweep over.
DISAGREEMENT_WEIGHT = 0.8

# Words that carry no schema meaning. Counting them as misses would make every
# question look uncovered.
_STOPWORD_TEXT = """
a an the of in on at to for from by with and or not is are was were be been
do does did how what which who whom whose when where why many much more most
less least there their they it its this that these those please list show
give tell find all any each every some no total number count average sum
maximum minimum highest lowest name names value values between among than
have has had can could would should will s t
"""

# Written as prose rather than a quoted list: sixty words in quotes is harder
# to read and harder to edit without introducing a typo.
STOPWORDS = frozenset(_STOPWORD_TEXT.split())  # noqa: SIM905


@dataclass(frozen=True)
class Verdict:
    decision: str
    confidence: float
    reasons: tuple[str, ...]
    # Kept so a threshold sweep can recompute decisions from stored
    # results without re-running anything.
    coverage: float = 1.0

    @property
    def answered(self) -> bool:
        return self.decision == ANSWER


def stem(token: str) -> str:
    """Strip a plural ending so `artists` matches a table named `artist`.

    Questions are written in plurals and schemas are usually named in the
    singular. Without this, "which artists are from Mali" scores zero
    vocabulary coverage against a schema containing an `artist` table, and
    the system abstains from questions it can answer perfectly well.

    Deliberately crude. A real stemmer would bring a dependency and a lot of
    behaviour, and the heuristic it feeds is now recorded rather than scored.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(question: str) -> list[str]:
    return [stem(token) for token in tokenize(question) if token not in STOPWORDS]


def vocabulary_coverage(question: str, schema_tokens: set[str]) -> float:
    """Fraction of the question's content words that appear in the schema.

    A blunt instrument, but it catches the case that matters: a question about
    something the database does not store at all.
    """
    tokens = content_tokens(question)
    if not tokens:
        return 1.0
    found = sum(1 for token in tokens if token in schema_tokens)
    return found / len(tokens)


def assess(
    question: str,
    *,
    schema_tokens: set[str],
    unresolved_concern: bool = False,
    query_failed: bool = False,
    agreement: float | None = None,
    clarify_below: float = DEFAULT_CLARIFY_BELOW,
    refuse_below: float = DEFAULT_REFUSE_BELOW,
) -> Verdict:
    """Score an answer's trustworthiness and decide what to do with it."""
    if query_failed:
        return Verdict(
            decision=REFUSE,
            confidence=0.0,
            reasons=("no query could be executed",),
            coverage=vocabulary_coverage(question, schema_tokens),
        )

    score = 1.0
    reasons: list[str] = []

    if agreement is not None and agreement < 1.0:
        score -= (1.0 - agreement) * DISAGREEMENT_WEIGHT
        reasons.append(
            f"repeated attempts agreed only {agreement:.0%} of the time"
        )

    if unresolved_concern:
        score -= UNRESOLVED_CONCERN_PENALTY
        reasons.append("the result still looked wrong after a second attempt")

    # Still computed and recorded, but no longer scored: it turned out not to
    # predict wrongness. Keeping the measurement makes it cheap to re-test on
    # a larger run without re-answering anything.
    coverage = vocabulary_coverage(question, schema_tokens)

    score = max(0.0, min(1.0, score))

    if score < refuse_below:
        decision = REFUSE
    elif score < clarify_below:
        decision = CLARIFY
    else:
        decision = ANSWER

    return Verdict(
        decision=decision, confidence=score, reasons=tuple(reasons), coverage=coverage
    )


def schema_vocabulary(table_names: list[str], column_names: list[str]) -> set[str]:
    """Every token appearing in the names of the tables and columns shown.

    Stemmed on the same rules as the question, so the two sides meet in the
    middle rather than one having to guess the other's pluralisation.
    """
    tokens: set[str] = set()
    for name in [*table_names, *column_names]:
        tokens.update(stem(token) for token in tokenize(name))
    return tokens
