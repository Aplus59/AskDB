"""Prompt construction and response parsing.

Kept apart from the loop so both can be tested without a model, and so the
wording can be changed without touching control flow.
"""

import re

FENCED = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

DRAFT = """\
You write SQLite queries against the schema below.

{schema}

Question: {question}
{hint}
Rules:
- Reply with one SQL query and nothing else. No explanation, no commentary.
- Use only the tables and columns shown above.
- SQLite dialect. Quote identifiers containing spaces with double quotes.
"""

REPAIR = """\
You write SQLite queries against the schema below.

{schema}

Question: {question}
{hint}
You previously wrote:
{sql}

The database rejected it with this error:
{error}

Rules:
- Reply with one corrected SQL query and nothing else.
- Fix the specific problem the error describes.
- Use only the tables and columns shown above.
"""


RECHECK = """\
You write SQLite queries against the schema below.

{schema}

Question: {question}
{hint}
You wrote this query, and it executed without error:
{sql}

However the result looks wrong: {concern}

Rules:
- Reply with one SQL query and nothing else.
- If the query was already right, reply with it unchanged.
- Use only the tables and columns shown above.
"""


def _hint(evidence: str) -> str:
    """Evidence is optional, and an empty label reads as a missing value."""
    return f"Hint: {evidence}\n" if evidence.strip() else ""


def draft_prompt(schema: str, question: str, evidence: str = "") -> str:
    return DRAFT.format(schema=schema, question=question, hint=_hint(evidence))


def repair_prompt(
    schema: str, question: str, sql: str, error: str, evidence: str = ""
) -> str:
    return REPAIR.format(
        schema=schema,
        question=question,
        hint=_hint(evidence),
        sql=sql,
        error=error,
    )


def recheck_prompt(
    schema: str, question: str, sql: str, concern: str, evidence: str = ""
) -> str:
    return RECHECK.format(
        schema=schema,
        question=question,
        hint=_hint(evidence),
        sql=sql,
        concern=concern,
    )


def extract_sql(response: str) -> str:
    """Pull the query out of a model response.

    Models wrap SQL in markdown fences regardless of instructions not to, and
    often add a sentence before it. Passing that text straight to SQLite
    produces a syntax error that looks like a model failure but is really a
    parsing bug, so this is handled explicitly rather than hoped away.
    """
    fenced = FENCED.search(response)
    text = fenced.group(1) if fenced else response
    return text.strip()
