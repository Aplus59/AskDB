# Schema retrieval baseline

BM25 over table documents, measured on BIRD Mini-Dev: 500 questions across 11
databases. All 500 were scored; none were skipped, so gold-table extraction
handled every query in the set.

## Results

| Query | recall@1 | recall@3 | recall@5 | recall@10 |
|-------|---------:|---------:|---------:|----------:|
| Question only | 0.108 | 0.564 | 0.760 | 0.972 |
| Question + evidence | **0.160** | **0.748** | **0.924** | 0.998 |

`recall@k` requires *every* table the gold query reads to appear in the top k.

## BIRD's evidence field is worth using

Appending the expert-written `evidence` field to the retrieval query lifts
recall@3 by 18.4 points and recall@5 by 16.4 points, at no cost — it is text
that ships with the question. It is used by default from here on.

## Two metrics that look better than they are

**recall@10 is uninformative on this dataset.** Nine of the eleven databases
have eight tables or fewer, so retrieving ten of them returns the entire
schema. A score of 0.998 mostly measures how small these schemas are.

| Database | Tables | Avg gold tables per question |
|----------|-------:|-----------------------------:|
| california_schools | 3 | 1.83 |
| thrombosis_prediction | 3 | 1.92 |
| toxicology | 4 | 2.00 |
| debit_card_specializing | 5 | 2.00 |
| card_games | 6 | 1.83 |
| european_football_2 | 7 | 1.82 |
| codebase_community | 8 | 2.00 |
| financial | 8 | 2.47 |
| student_club | 8 | 2.06 |
| superhero | 10 | 2.54 |
| formula_1 | 13 | 2.15 |

**recall@1 is capped by the data, not by the retriever.** Questions need 2.1
tables on average, and the metric requires all of them, so a top-1 result can
only ever satisfy the single-table questions. A low recall@1 here is close to
meaningless as a quality signal.

That leaves **recall@3 and recall@5** as the metrics that carry information on
this dataset.

## What this changes

Table-level retrieval is not the bottleneck on Mini-Dev. At recall@5 = 0.924
with evidence, the correct tables are already in the prompt for nine questions
in ten, and the remaining failures are concentrated in `formula_1`, the only
database with more than ten tables.

Adding dense embeddings was the planned next step. Measurement says not yet:
there is little headroom to buy, and the cost and latency would be paid on
every query. The baseline existed precisely to catch this, and it did.

Two places where retrieval will matter and this result does not transfer:

- **Spider 2.0**, whose enterprise schemas are far larger than anything here.
- **Column selection.** These databases have few tables but wide ones —
  `california_schools` has three tables and one of them carries around a
  hundred columns. Tokens and model confusion live at the column level, not
  the table level.

## Reproducing

```bash
python scripts/evaluate_retrieval.py
python scripts/evaluate_retrieval.py --with-evidence
```
