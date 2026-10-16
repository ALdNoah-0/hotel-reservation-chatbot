# Evaluation: measured results and limitations

Measured on 2026-10-16 with Python 3.14, scikit-learn 1.9.1, and Streamlit 1.65.0. Every number below except the hand-graded spot checks is reproducible:

```bash
python scripts/evaluate.py
python -m pytest
```

`scripts/evaluate.py` writes every prediction, routing method, and per-class metric to `evaluation/results.json`, and the current full-pipeline mistakes on the regression benchmark to `evaluation/errors.csv`. The app's **How the NLP works** tab reads the same `results.json`.

## Summary

| Test set | Messages | Classifier alone | Full pipeline |
|---|---:|---:|---:|
| Independent held-out intents | 60 | 95.00% | 96.67% |
| Independent held-out FAQ questions | 37 | 72.97% (retriever alone) | 100.00% |
| Notebook regression benchmark (casual phrasing) | 96 | 67.71% | 100.00%* |

\* The rules were written after inspecting this benchmark, so the 100% is a regression check, not independent evidence. See below.

"Classifier alone" is the TF-IDF + logistic regression model with no rules. "Full pipeline" is what the chatbot actually uses: priority rules first, then the classifier and the FAQ retriever with fixed confidence thresholds. Ordinary accuracy counts abstentions ("please rephrase") as errors.

## Test sets

| File | Contents | Used for training or tuning? |
|---|---|---|
| `data/intents.json` | 849 labelled training messages, 18 intents | Training |
| `data/regression_holdout.json` | 96 casual-language messages from the original notebook | Inspected while writing rules and used as the dev set for classifier tuning; not independent |
| `data/independent_test.json` | 60 intent messages and 37 FAQ questions, written before the training-data expansion | Never used for training or tuning |

Three generic greetings in the independent set duplicated original notebook training text. An automated overlap test found them and they were removed before the scores above; nothing was relabelled. `tests/test_mvp.py` fails if any evaluation message ever appears in the training data, and `scripts/augment_training.py` refuses to add one.

## Notebook regression benchmark (96 messages)

| Engine | Correct | Accuracy | Macro F1 | Abstentions |
|---|---:|---:|---:|---:|
| Original notebook model (saved historical result) | 45 | 46.88% | 0.4687 | not reported |
| Previous classifier, before the MVP update | 39 | 40.62% | 0.5018 | 42 |
| **MVP classifier alone** | 65 | 67.71% | 0.6992 | 6 |
| MVP full pipeline | 96 | 100.00% | 1.0000 | 0 |

The MVP classifier answers 93.8% of these messages, and 72.2% of its answers are correct (macro precision 0.779, macro recall 0.677). It still misses about a third of these very casual phrasings on its own; the rules catch them in the full pipeline.

## Independent held-out set

| Engine | Intent accuracy (60) | Macro F1 | FAQ topic accuracy (37) |
|---|---:|---:|---:|
| MVP classifier alone | 95.00% (57/60) | 0.9513 | — |
| MVP full pipeline | 96.67% (58/60) | 0.9738 | 100.00% (37/37) |
| FAQ retriever alone | — | — | 72.97% (27/37) |

Full-pipeline misses (left unfixed so the set stays clean):

- "can I see what rooms are open on january 5": expected `check_availability`, got `room_types_inquiry`
- "tell me about your suites": expected `room_types_inquiry`, got `amenities_inquiry`

Classifier-only misses: "what is the cheapest room you have" (→ `room_types_inquiry`), "how do I find you" (→ `out_of_scope`), and "if I cancel will I be charged" (→ `cancel_booking`). The rules route all three correctly in the full pipeline.

Caveat: this set is small and was written by the same author as the new training examples, so its phrasing may be closer to the training data than real guests' messages are. The 67.71% on the casual-language benchmark is the more conservative estimate of how the classifier handles unfamiliar wording.

## What changed in the classifier

The model is still word 1–2-gram + character 3–5-gram TF-IDF features feeding a class-balanced logistic regression. Each change was compared with 5-fold cross-validation on the training data and on the regression benchmark as a dev set; the independent set was not consulted.

| Change | Effect on the regression benchmark (classifier alone) |
|---|---|
| Contraction and slang expansion ("gonna" → "going to") plus entity masking (dates, room names, numbers, emails, and booking references become `xdate`, `xroom`, `xnum`, `xemail`, `xref`) | Accuracy before abstention 63.5% → 68.8% |
| Suffix stemming | Rejected: accuracy before abstention fell to 64.6% |
| 273 new hand-written, varied training examples, adding the `view_booking` and `cancellation_policy` classes | Fewer abstentions on unknown vocabulary |
| C = 10 (was 6); abstain below probability 0.175 or a top-two margin of 0.02 (was 0.35 / 0.08). The threshold maximises out-of-fold utility of +1 per correct answer, 0 per abstention, −1 per wrong answer. | 40.62% → 67.71% accuracy; 42 → 6 abstentions |

In the dialogue manager, a classifier result below 0.55 never starts a booking, change, or cancellation; the bot asks a clarifying question instead. On hand-labelled probe messages, every wrong classifier route scored 0.52 or lower, and every correct held-out classifier route scored 0.56 or higher.

## FAQ retrieval

39 hotel topics are matched first by precise patterns, then by TF-IDF cosine similarity against 212 example questions. A retrieved topic counts only if it shares at least one content word with the message, and generic hotel words ("stay", "hotel", "nights") do not count as evidence. Without these guards, "do you accept bitcoin" retrieved the pet policy ("do you accept animals") and "how long can I stay" retrieved it via "can my cat stay". The guards cost some retriever-only recall (75.68% before the guards, 72.97% now) without changing the full-pipeline FAQ score.

## Hand-graded spot checks on new questions

These are manual judgements, not reproducible metrics.

| Batch | Questions | Answered well, or honest "no information" | Partly right | Wrong |
|---|---:|---:|---:|---:|
| Batch 2, before the fixes below | 60 | about 38 | about 7 | about 15 |
| Batch 3, after the fixes, not used for tuning | 35 | 25 | 5 | 5 |

Batch 2 errors led to new housekeeping, noise, medical, and feedback topics; wider visitor, airport drop-off, payment-currency, receipt, event, ATM, and key-card patterns; a narrower "talk to a human" rule; and honest "no information" replies for items the hotel does not list. Of the five batch 3 misses, two were fixed afterwards (a capacity question for five people, plural "bathtubs"). The low-confidence clarify gate now turns the others (surfboard storage, COVID rules) into clarifying questions rather than wrong actions. Vending machines remain unhandled.

Expect roughly one in seven genuinely new questions to need rephrasing or to get a partial answer.

## Automated and manual verification

- **239 automated tests pass.** They cover date, guest, night, and week parsing; overlapping and adjacent stays, concurrent attempts for the last room, idempotent confirmation, wrong-email lookups, and failed modifications that leave the original booking intact; multi-turn conversations, negation ("do not cancel"), corrections, and confirmation boundaries; every dropdown question returning a real answer; FAQ routing; the clarify gate; and Streamlit AppTest rendering, form booking, and lookup/cancellation.
- **Live UI walkthrough on a throwaway database:** a chat booking with a special request, Confirm, the receipt and downloaded summary (contents checked), chat status lookup, and the Manage tab lookup, change (2 → 3 nights, PHP 27,000), and cancellation all worked with no errors.

## Limitations

- **Fictional hotel.** Hotel Becerel and every fact in `data/hotel.json` (address, contacts, hours, fees, policies) are demo content. Replace them with verified facts before any real use.
- **Small, author-written evaluation sets.** No user study was run and no real guest messages were collected. The independent set has 60 intent and 37 FAQ messages.
- **The regression benchmark is exposed.** Its 100% full-pipeline score shows the rules cover it, not that the system generalises.
- **Scores are routing confidences, not calibrated probabilities**, and rule matches report 1.0 by construction.
- **English only, one room per reservation, 1–30 nights.** Relative ranges such as "next weekend" ask for exact dates instead of guessing.
- **No payments, real inventory, or staff messaging.** Reservations live in a local SQLite file, and the bot cannot contact reception.
- **Single-machine demo.** The database lives on the machine running the app, and the classifier is retrained each time the app starts.
