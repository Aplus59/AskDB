"""Deciding whether to answer at all.

Most text-to-SQL systems have two outcomes: an answer, or a wrong answer
presented as an answer. A confidently wrong number is worse than no number,
because nobody knows to check it. This adds a third outcome — ask, or decline.

The score is assembled from signals the agent has already produced, so
abstention costs no extra model call. That matters: a system that spends a
call deciding whether to spend a call has doubled its cost before answering
anything.

Penalties, and why each one:

- An unresolved concern. The self-check flagged the result, a recheck was
  spent, and it still looks wrong. That is the strongest available evidence
  that the answer is not trustworthy.
- Repairs. A query that needed correcting was wrong once already.
- Vocabulary coverage. When the content words of a question find nothing in
  the schema, the database probably cannot answer it, and the model will
  produce plausible SQL over the wrong columns rather than say so.
"""

from dataclasses import dataclass

from askdb.retrieval.documents import tokenize

ANSWER = "answer"
CLARIFY = "clarify"
REFUSE = "refuse"

DEFAULT_CLARIFY_BELOW = 0.6
DEFAULT_REFUSE_BELOW = 0.3

UNRESOLVED_CONCERN_PENALTY = 0.4
REPAIR_PENALTY = 0.15
LOW_COVERAGE_PENALTY = 0.3
COVERAGE_FLOOR = 0.25

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
    behaviour, to improve a heuristic whose output is one input among four.
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
    repairs: int = 0,
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

    if repairs:
        score -= REPAIR_PENALTY * repairs
        reasons.append(f"the query needed {repairs} correction(s)")

    coverage = vocabulary_coverage(question, schema_tokens)
    if coverage < COVERAGE_FLOOR:
        score -= LOW_COVERAGE_PENALTY
        reasons.append(
            f"only {coverage:.0%} of the question's terms appear anywhere in the schema"
        )

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
