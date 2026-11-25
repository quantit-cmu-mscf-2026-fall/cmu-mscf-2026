# Jev (TypeSafe): what it is good for, and where it fits in the factor pipeline

**Research note, 2026-11-24.** Written after our alignment-judge calibration run
(`research/judge_calibration.py`), where Jev (`jev-latest`, served as
`jev-1.14.0`) could not tell aligned factor formulas from misaligned ones. Facts
from TypeSafe's own pages are marked **[official docs]**. Results from papers by
other people are marked **[third-party evaluation]**. Almost all of those papers
tested `jev-1.13.0`, the version before the one we called.

## Summary and recommendation

**Don't use Jev to check whether a formula matches its hypothesis.** TypeSafe's
own list of weak spots covers that task: reading symbols and numbers,
questions that need several steps of reasoning, and literal reading of
negations ([model jaggedness](https://docs.typesafe.ai/model-jaggedness)).
Independent studies agree. Jev does well on "short decisions about the gist of
short English text" and badly on multi-step or derivation-style checks
([2609.37647](https://arxiv.org/abs/2609.37647),
[2609.32160](https://arxiv.org/abs/2609.32160)). Keep Claude, or better,
code plus Claude, for alignment.

**Where Jev may be worth a cheap trial:**

1. **Classifying the hypothesis text**, with code doing the comparison. Jev
   reads only the prose and answers: which direction does it claim, over what
   time scale, using which data fields. Code works out the same properties
   from the parsed formula and compares the two. This gives Jev the kind of
   judgment its documentation says it handles well, and leaves the symbolic
   part to the parser.
2. **Paperlog pre-screening and tagging.** Use one Score per relevance
   dimension, one Noul per tag and one Choice for look-ahead bias. Run it as a
   cheap first pass or a cross-check next to the existing LLM screen, not as a
   replacement. Summaries and translations still need a generative model.

Both trials cost cents (price: $0.042 per million input tokens). Each needs a
labelled split, so that thresholds are chosen on one half and tested on the
other. First, compute the AUROC of the probabilities we already have from the
failed run. If AUROC is near 0.5, the ranking is broken and no threshold can
fix it.

## What our run measured

From the cached answers of the full calibration run (`research/judge_calibration.py`,
60 crsp-v3-50p proposals plus 232 items built to fail one check), Jev's probabilities
rank aligned above misaligned items only weakly. The AUROC is the chance a random
aligned item scores above a random misaligned one: 0.5 is chance, 1.0 is perfect.

| Check | AUROC | Items expected to pass / fail |
|---|---|---|
| variables | 0.74 | 175 / 57 |
| direction | 0.69 | 172 / 60 |
| horizon | 0.73 | 177 / 55 |

On the same items Claude Sonnet 5.5 let through 0 of 57 field swaps, 0 of 60 negated
formulas and 2 of 55 window rescales. So no threshold fixes Jev on this task. The
questions were written as in that run; the redesigned trials below are untested.

## 1. What tasks Jev is built for

**[official docs]**

- Jev is a "System One" model. It does not generate text. It returns a typed
  answer with probabilities for each question about a `state`
  ([System One](https://docs.typesafe.ai/concepts/system-one)). The docs ask
  for "a judgment a knowledgeable person makes in a second given the right
  context". A request to "analyze ... and determine" is "a signal to break the
  task into small questions"
  ([Primitives](https://docs.typesafe.ai/primitives)).
- **Input type.** "Jev reads natural-language text, sent as a string, a JSON
  object, or an array of text values", plus PNG/JPEG images
  ([Models](https://docs.typesafe.ai/models)).
- **Symbols and numbers.** No page addresses formulas or code as input
  directly. The closest guidance is in
  [Model jaggedness](https://docs.typesafe.ai/model-jaggedness):
  - "Jev will perform better on semantic representations than numeric." Hex
    colour codes do worse than colour names.
  - "Questions about high-level programming languages will perform better than
    questions about low level assembly."
  - Jev "is not a calculator", "does not count reliably", and "reads dates as
    text, not as ordered quantities".
- **Multi-step reasoning.** "It may struggle with tasks that require additional
  levels of indirection ... A question about a property of a property or
  something that requires multiple hops of reasoning costs accuracy." It is
  "quite literal": "Scoping words, negations, and implied conditions are read at
  face value" ([Model jaggedness](https://docs.typesafe.ai/model-jaggedness)).
- **Long context.** The default limits are 64k tokens per request and 32k per
  question. An opt-in header raises both to 262k, but the docs say it is
  "slower and less accurate" ([Models](https://docs.typesafe.ai/models)).
  "Accuracy falls as the state grows with content unrelated to the decision"
  ([Model jaggedness](https://docs.typesafe.ai/model-jaggedness)).
- **Examples in the request.** There is no separate few-shot feature. The docs
  suggest putting examples in the state or in named fields of structured
  `instructions`/`criteria`
  ([State](https://docs.typesafe.ai/concepts/state),
  [How to build](https://docs.typesafe.ai/concepts/how-to-build-with-system-one)).
  The same weights serve every account. Jev is not fine-tuned on customer data
  ([Models](https://docs.typesafe.ai/models)).

**[third-party evaluation]**

- **Benchmark of 37 datasets, zero-shot, `jev-1.13.0`**
  ([2609.37647](https://arxiv.org/abs/2609.37647), §4.1):
  - Strongest "where a decision rests on the overall gist of a short text in a
    widely spoken language": IMDB 96.5%, BoolQ 91.3%, ARC-Challenge 97.7%,
    CLINC150 89.5% with 151 options.
  - Weaker on adversarial inference (ANLI round 3: 67.7%), grounding checks
    (LLM-AggreFact balanced accuracy 78.6%), legal judgment (AGB-DE F1 0.204)
    and rubric scoring of generated text.
  - The authors excluded code and arithmetic tasks, following TypeSafe's
    advice (§3.2). So this benchmark does not test symbolic input at all.
- **Review of 28 early papers**
  ([2609.32160](https://arxiv.org/abs/2609.32160), Finding 2 and §7):
  - Jev trails stronger models on harder evaluations, for example by 14.6
    points on JudgeBench and up to 16.4 points on graded rubric criteria.
  - The corpus "shows weaknesses on multi-step arithmetic" and "derivation
    checking". One plausible cause: the interface "provides no explicit
    iterative or generated intermediate computation".
  - Decisions can be steered by added context: optimized answer-preserving
    additions redirect 61.4% of initially correct decisions.
  - These are the review's summaries of other papers. I did not read those
    papers.
- **Direction of a stance**
  ([2610.11901](https://arxiv.org/abs/2610.11901), §5 and App. C). This is the
  closest analogue to our direction check.
  - On single English texts (VAST), Jev matched GPT-5.6.
  - On conversational stance (ZS-CSD), it made 373 favor-vs-against errors,
    against 35 for GPT-5.6 and 101 for Claude Sonnet 5. It did as well as they
    did at telling neutral from non-neutral.
  - The authors relate the failures to "understanding reply relationships and
    stance direction": polarity that has to be worked out through another
    object.
- **Examples in the state, on numeric input**
  ([2610.00376](https://arxiv.org/abs/2610.00376)). The input was packet
  sizes and timings as JSON numbers.
  - 40 labelled examples in the state raised accuracy from 9.8% to 28.4%, and
    150 examples reached 34.5% (one week of data).
  - Tree ensembles trained on 8,000 rows reached about 70%.
  - So examples help, but Jev does not learn a numeric mapping from context.
- **Decomposition**
  ([2610.08829](https://arxiv.org/abs/2610.08829), §5). Splitting emotion,
  sarcasm and humour classification into atomic Noul judgments (the docs'
  main recommendation) "did not consistently improve" macro-F1: 62.0% against
  62.9% for one direct question. Decomposition is not guaranteed to help.

**What this means for formulas.** No source tests Jev on structured formulas.
But every relevant documented weakness applies to our task:

- symbols instead of words;
- window lengths that must be turned into time scales (arithmetic);
- a leading unary minus, plus `rank`, plus division, that set the sign
  (negation, read literally);
- "does a higher value mean what the hypothesis says should outperform", which
  asks about a property of a property.

Our result (negated formulas passed at threshold 0.1, horizon P(yes) about 0.18
on every item) is what those documented limits predict.

## 2. How to design questions

**[official docs]**

**Choosing the type**
([Primitives](https://docs.typesafe.ai/primitives)):

- **Noul** for a clean yes/no. It returns one probability with no separate
  confidence.
- **Choice** for one of several unordered options. It returns `probabilities`
  and `confidence`. The OpenAPI schema allows up to 255 options
  ([openapi.json](https://api.typesafe.ai/openapi.json)), and one cookbook
  says "a Choice works reliably up to roughly 240 options"
  ([SEC cookbook](https://docs.typesafe.ai/cookbooks/classification_using_confidence)).
- **Score** for ordered levels (1 to 10 in the schema).
- **Box** for image regions.
- A Choice always picks an option. Rather than adding an `other` option, which
  "competes with your real options for probability", ask a *companion Noul*:
  "does any option fit?"
  ([Companion Noul](https://docs.typesafe.ai/patterns/companion-noul)).

**Wording**
([Noul](https://docs.typesafe.ai/primitives/noul),
[Model jaggedness](https://docs.typesafe.ai/model-jaggedness)):

- Ask one condition per Noul. "If a question has two conditions ... the value
  means less. Ask two Nouls and combine them in code."
- Phrase it so that a high value means yes.
- `criteria.true` and `criteria.false` are optional. "Try your questions with
  and without `criteria` and keep whichever gives better answers."
- Criteria that contradict the instruction hurt. A Noul "where `true` maps to no
  and `false` maps to yes will perform worse."
- "Aim for instructions which are easy for the average person to read and
  understand."

**Structure** ([Advanced](https://docs.typesafe.ai/primitives/advanced),
[How to build](https://docs.typesafe.ai/concepts/how-to-build-with-system-one)):

- `instructions`, choice options, score levels and `criteria.true/false` can
  each be a string, an object or an array.
- Use an object to separate the question from supporting data, for example
  `{"question": ..., "compare": [...], "focus": ...}`. Criteria can be objects
  with `what`, `not_for` and `examples` fields.

**State** ([State](https://docs.typesafe.ai/concepts/state),
[Primitives](https://docs.typesafe.ai/primitives)):

- "Use an object for most requests so each part of the state has a descriptive
  name."
- Point questions at fields with backticked paths, such as `ticket.message`.
- Send only what the question needs.

**Several questions per call**
([Primitives](https://docs.typesafe.ai/primitives),
[Parallel questions](https://docs.typesafe.ai/cookbooks/parallel_questions)):

- Questions are "evaluated independently". "One question's answer is not hidden
  context for another." There is no cap on the number, only the context limits.
- TypeSafe measured 13 questions over one ~54k-character document, all in one
  request versus one per request. Batching left the answers unchanged (run-to-run
  spread ≤ 0.015) and was 12.2 times cheaper and 11.8 times faster.
- So our three questions per call did not interfere with each other.

**Closest official examples.** Two cookbooks are close to "does A match B":

- [Double-checking citations](https://docs.typesafe.ai/cookbooks/citation_check)
  uses one Choice with three verdicts (supports / contradicts / says nothing)
  on a claim and a source section. Code removes fabricated quotes first, by
  string match.
- [Verifying LLM answers](https://docs.typesafe.ai/cookbooks/verify_llm_outputs)
  uses three Nouls (answerable / declines / supported).
  - On 1,000 multi-hop MuSiQue answers, a risk threshold of 0.2 passed 73% of
    answers, and 95.7% of those were correct, against 88.5% overall.
  - It still let through 18 answers to a related question instead of the one
    asked.

Both compare *natural-language* statements.

**Why our design likely failed, checked against the docs.** See
`research/judge_calibration.py`.

| What we did | What the docs say |
|---|---|
| Each Noul's instructions were the Claude judge's system prompt ("You check whether ...") plus the question | Ask "one snap judgment per question" ([Primitives](https://docs.typesafe.ai/primitives)) |
| `direction`: "does a higher value mean what the hypothesis says should outperform?" | Property-of-a-property questions cost accuracy ([jaggedness: indirection](https://docs.typesafe.ai/model-jaggedness)) |
| `horizon`: compare windows in trading days to a stated horizon, and "answer yes if the specification states none" | Keep arithmetic and date comparison in code. Don't hide a conditional default inside one Noul; split it ([jaggedness](https://docs.typesafe.ai/model-jaggedness), [Noul](https://docs.typesafe.ai/primitives/noul)) |
| State was one string: five hypothesis fields, formula, rationale and the full grammar notes | Use an object with named fields. Leave out content the question doesn't need, because it acts as a distractor ([State](https://docs.typesafe.ai/concepts/state), [jaggedness](https://docs.typesafe.ai/model-jaggedness)) |
| Formula as written: `-(rank(ts_mean((volume / shares), 252)))` | "Semantic representations" do better than numeric or low-level ones ([jaggedness](https://docs.typesafe.ai/model-jaggedness)) |
| Threshold 0.5 | Start at 0.7 / 0.3 and send the middle band elsewhere ([Noul](https://docs.typesafe.ai/primitives/noul)). Single Nouls run slightly under-confident, so true positives often score below 0.5 ([2609.37647](https://arxiv.org/abs/2609.37647) §4.3, **[third-party evaluation]**) |
| `model: jev-latest` | Aliases move. Pin the version you tuned thresholds on ([Models](https://docs.typesafe.ai/models)) |

The under-confidence finding explains why 56 of 60 good items were rejected at
0.5. It does not explain why negated formulas passed even at 0.1. That points
to a failure of ranking, not of threshold.

A small code issue: our retry loop retries 429 and 5xx errors but not
`529 Overloaded`, which the
[errors page](https://docs.typesafe.ai/api/errors) lists as retryable.

**Should we translate the formula into English first?** The docs support it in
part. Semantic text does better than numeric or symbolic text, and both
comparison cookbooks compare plain-language statements. But it only removes
the symbol-reading problem. Direction still takes several steps unless the
English rendering states it outright ("buys the stocks with the *lowest*
turnover"). And if code can write that sentence, code already knows the
direction and can compare it with a label that Jev extracts from the
hypothesis. The cleaner split is:

- **Code** extracts formula properties: the fields used, the longest window
  turned into a time-scale bucket, and the sign of the signal with respect to
  its main field. The sign is easy to derive for monotone operators
  (`rank`, `zscore`, `ts_mean`, `ts_sum`, unary minus, the denominator of a
  division). It is not for others (`ts_std`, `ts_corr`, products), which go to
  the LLM judge.
- **Jev** classifies the *hypothesis prose* into the same buckets.
- **Code** compares the two.

Experiment 2 below still tests the translate-first idea directly, because it
costs almost nothing.

## 3. Where Jev could fit in the pipeline

Ranked by how well the evidence supports each use.

| Use | Fit | Evidence |
|---|---|---|
| **Relevance screening of paper titles and abstracts** (paperlog) | Good, as a first stage | Topic and gist classification of short English text is Jev's strongest area ([2609.37647](https://arxiv.org/abs/2609.37647) §4.1). The vendor documents a "screen with a Noul first" pattern ([Noul screen](https://docs.typesafe.ai/patterns/noul-screen)). Our relevance rubric mixes two dimensions (METHODS or FINDINGS, whichever is higher). The docs say low Score confidence "often means the levels are ambiguous, multi-dimensional" ([Confidence](https://docs.typesafe.ai/confidence)), so ask two Scores and take the maximum in code. Some papers are in Chinese: 1.14 claims "better multilingual support" ([changelog](https://docs.typesafe.ai/changelog)), while third parties saw drops on lower-resource languages and on Chinese conversational stance ([2609.37647](https://arxiv.org/abs/2609.37647), [2610.11901](https://arxiv.org/abs/2610.11901)) |
| **Tagging papers into our 18-tag taxonomy** | Plausible, needs tuned thresholds | Multi-label tagging is one Noul per tag. In the 37-dataset benchmark, multi-label Nouls over-predicted "yes" (mean 0.209 against an observed rate of 0.041, ECE 0.168) but ranked well (AUROC 0.873 to 0.994). Per-tag thresholds tuned on 1,000 examples raised micro-F1 from 0.50 to 0.75 on UNFAIR-ToS, with a median tuned threshold of 0.86 on GoEmotions ([2609.37647](https://arxiv.org/abs/2609.37647) §4.3). "Fine-grained or noisy label sets" degrade accuracy (abstract). Some of our tags overlap (`backtest_overfitting` / `multiple_testing`; `factor_models` / `return_anomaly`). Large-taxonomy Choice works in the vendor's demos ([hierarchical](https://docs.typesafe.ai/cookbooks/hierarchical_classification), [SEC 75-way](https://docs.typesafe.ai/cookbooks/classification_using_confidence)), but those are vendor demos on 3 and 60 items |
| **Hypothesis-text properties** (direction, horizon and variables *as stated in prose*) | Plausible, untested | A short English text with a fixed label set is in Jev's documented range. Direction is the risky part: the stance paper found polarity errors when polarity has to be worked out through another object ([2610.11901](https://arxiv.org/abs/2610.11901)). Phrase it so polarity is stated directly ("stocks with higher X earn higher returns"). Use a companion Noul for "not stated" |
| **Archetype labelling of hypothesis text** (e.g. `kind`: market vs method, or a finer archetype list) | Plausible | This is a Choice with contrastive criteria, the documented pattern ([Advanced](https://docs.typesafe.ai/primitives/advanced)). Watch option order: "Jev leans toward the option that comes first" ([jaggedness](https://docs.typesafe.ai/model-jaggedness)), so test with the options reordered |
| **Formula-to-hypothesis verification** (our failed run) | Poor | See section 1. No source shows Jev reading symbolic formulas, and the vendor lists every sub-skill involved as a weakness |

What stays with an LLM in every case: summaries, translations, structured
extraction (methods, datasets, markets) and anything else that produces text.
The docs say this themselves: "If you really need to generate text... there are
other models for that" ([jaggedness](https://docs.typesafe.ai/model-jaggedness)).

Cost is not a decisive argument either way.

- **[official docs]** Jev bills input tokens only
  ([Models](https://docs.typesafe.ai/models)).
- **[third-party evaluation]** One study found Jev's per-decision cost *higher*
  than GPT-6 Luna's on six of seven tasks at OpenAI batch prices, because Jev
  bills the question and every option as input (1.2 to 3.3 times the input
  tokens) ([2610.06625](https://arxiv.org/abs/2610.06625), Table 2 and
  the introduction). Jev's clear
  advantage there was speed.

## 4. Calibration and thresholds

**[official docs]**

- Jev is trained with "RLCD" (reinforcement learning for calibrated decisions),
  so that "outcomes assigned a probability of 0.8 should occur about 80% of the
  time". The docs add that this describes "groups of predictions, not a
  guarantee about any single answer"
  ([AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer)).
- Noul thresholds: "A good starting point is 0.7 for yes and 0.3 for no". Send
  the 0.3 to 0.7 band to its own path. "Avoid fitting thresholds closely to a
  handful of test cases"
  ([Noul](https://docs.typesafe.ai/primitives/noul)).
- For a Choice, put the threshold on the chosen option's probability. For a
  Score that depends on one level, use that level's probability. `confidence`
  is a fixed transform of those values. "Test thresholds by plotting confidence
  against accuracy on your data"
  ([Confidence](https://docs.typesafe.ai/confidence),
  [How to build](https://docs.typesafe.ai/concepts/how-to-build-with-system-one)).
- "If you have tuned confidence thresholds against one version, send that
  version's ID instead of the alias"
  ([Models](https://docs.typesafe.ai/models)).
- Vendor tests:
  - Self-consistency on 40 borderline insurance claims: a claim's least stable
    decision changed 3.8% of the time on repeat, against 6.2% to 34.8% for the
    LLM settings tested
    ([self-consistency cookbook](https://docs.typesafe.ai/cookbooks/consistency_noul_cookbook)).
  - In the SEC cookbook, answers with `confidence` ≥ 0.9 were right 93% of the
    time, and the rest 50%
    ([SEC cookbook](https://docs.typesafe.ai/cookbooks/classification_using_confidence)).
  - These are vendor measurements on small, vendor-chosen sets.

**[third-party evaluation]**

- **Benchmark of 37 datasets**
  ([2609.37647](https://arxiv.org/abs/2609.37647) §4.3):
  - Choice probabilities are well calibrated: pooled ECE 0.028 over 279,925
    answers.
  - Single Nouls are slightly under-confident (ECE 0.052, mean 0.465 against a
    yes-rate of 0.518).
  - Multi-label Nouls over-predict yes (ECE 0.168).
  - A fixed 0.5 threshold "often turns this good ranking into mediocre
    decisions". In prompt-injection detection, recall was 50% at AUROC 0.982.
  - Thresholds tuned on a small training sample recover much of the gap.
- **Political-science replications**
  ([2610.06625](https://arxiv.org/abs/2610.06625)). Asked once per question,
  Jev was better calibrated than GPT-6 Luna's token probabilities, but "not
  consistently better than Qwen3.8-27B's". Pairwise comparisons showed order
  effects that averaging both orders removed.
- **Review of 28 papers**
  ([2609.32160](https://arxiv.org/abs/2609.32160), Finding 4 and Finding 5),
  citing other papers that I did not read:
  - "Raw calibration is inconsistent." In one study, held-out recalibration
    cut calibration error 3.3-fold.
  - Thresholds chosen on one set "do not hold up on held-out items for every
    fallback model".
  - High confidence does not rule out serious errors (wrong tool calls at mean
    confidence 0.778).
  - Probabilities come on a 0.01 grid, which limits resolution for rare labels.
- **Open replications.** Open LLMs read through answer-letter logits and
  temperature-scaled matched Jev's composite accuracy-and-calibration score
  (79.6 against 80.1), and beat it after LoRA fine-tuning
  ([2610.17205](https://arxiv.org/abs/2610.17205);
  [2610.02076](https://arxiv.org/abs/2610.02076) reaches a similar
  conclusion). So calibrated probabilities are not unique to Jev.

**How to choose thresholds for us:**

1. Pin `jev-1.14.0`.
2. On labelled items, report AUROC per question first. If AUROC is low, stop:
   no threshold will help.
3. Split the 60 proposals, with all items derived from each proposal, into a
   tuning half and a test half. Pick thresholds on the tuning half, report on
   the test half, and use a three-way band (pass / review / fail) rather than
   one cut.
4. Re-check whenever the version changes.

## 5. Cost, latency, rate limits, context

| Item | [official docs] | [third-party evaluation] |
|---|---|---|
| Price | $0.042 per million input tokens ($42 per billion); output free ([Models](https://docs.typesafe.ai/models)) | 346,009 requests (about 630 tokens each) cost $9.15 ([2609.37647](https://arxiv.org/abs/2609.37647) §4.7) |
| Latency | "Most calls complete in about 100 ms" ([How to build](https://docs.typesafe.ai/concepts/how-to-build-with-system-one)); "about 150ms" ([Use cases](https://docs.typesafe.ai/concepts/use-case-map)) | Mean 0.36 s at 32 concurrent requests ([2609.37647](https://arxiv.org/abs/2609.37647)); 0.09 to 0.27 s per decision ([2610.06625](https://arxiv.org/abs/2610.06625)); median 250 ms ([2610.08829](https://arxiv.org/abs/2610.08829)); median about 0.72 to 0.77 s via OpenRouter ([2610.00376](https://arxiv.org/abs/2610.00376)); most studies 0.15 to 0.45 s ([2609.32160](https://arxiv.org/abs/2609.32160)) |
| Rate limits | Per organization, in tokens per second and requests per second. Actual numbers are only shown in the developer console. Over the limit returns `429`; overload returns `529` ([Models](https://docs.typesafe.ai/models), [errors](https://docs.typesafe.ai/api/errors)) | One study ran at up to 1,100 requests per minute with 32 connections and was limited by the rate limit, not by the service ([2609.37647](https://arxiv.org/abs/2609.37647) §3.5) |
| Context | 64k tokens per request (state plus all questions), 32k per question (state plus one question). Opt-in header `TypeSafe-Danger-Zone-Long-Context: true` raises both to 262k, at the same price but "slower and less accurate" ([Models](https://docs.typesafe.ai/models)) | One of 346,010 requests went over the limit ([2609.37647](https://arxiv.org/abs/2609.37647)) |
| Request rules | Body ≤ 32 MiB; nesting ≤ 64 levels; finite numbers only (send exact large values as strings); gzip/zstd accepted ([errors](https://docs.typesafe.ai/api/errors)) | — |
| Question limits | No cap on the number of questions. Choice up to 255 options; Score 1 to 10 levels ([openapi.json](https://api.typesafe.ai/openapi.json)) | — |
| Data | Not trained on customer requests. Zero data retention is available for enterprise customers ([Models](https://docs.typesafe.ai/models), [Legal](https://docs.typesafe.ai/legal)) | — |

For scale: our 292-item run, at roughly 1,000 to 2,000 input tokens per
request, is about 0.5M tokens, or about $0.02.

## A redesigned Jev experiment for our pipeline

All three experiments:

- pin `jev-1.14.0`;
- use object states with named fields;
- ask short, single-judgment questions;
- leave arithmetic and the final comparison to code.

The field descriptions and tag definitions below are drafts for the team to
review. They are not checked against `capstone/factors/tree.py` or
`research/paperlog/config.yaml`.

### Experiment 1: Jev reads the hypothesis, code reads the formula

Send one request per (hypothesis, formula) pair, the same 292 items. Code
supplies the formula's main field in plain words. Jev never sees the formula.

**Code does:**

- parse the formula;
- list the fields it uses;
- turn the longest window into a time-scale bucket (21 trading days is about
  1 month);
- derive the sign with respect to the main field when the operators allow it.

**Jev answers, from `specification` only:**

- whether the hypothesis mentions each field (one Noul per field);
- the direction it claims, plus a companion Noul for "not stated";
- the time scale it claims, plus a companion Noul for "not stated".

**Pass rules, applied in code:**

- **direction** passes when `direction_stated` ≥ τ and the Choice matches the
  formula's sign.
- **horizon** passes when `horizon_stated` < τ′, or when the bucket matches.
- **variables** passes when the set of fields with P ≥ τ″ matches the
  formula's fields, up to listed equivalents.

```json
{
  "model": "jev-1.14.0",
  "state": {
    "specification": "Stocks with persistently low share turnover over the past year earn higher subsequent returns than high-turnover stocks, consistent with a liquidity premium.",
    "signal_field": "share turnover: shares traded per day divided by shares outstanding"
  },
  "questions": {
    "direction_stated": {
      "type": "noul",
      "instructions": "Does `specification` say whether stocks with more or less of `signal_field` should earn higher returns?"
    },
    "direction": {
      "type": "choice",
      "instructions": "According to `specification`, which stocks should earn higher returns?",
      "criteria": {
        "more_outperforms": {"what": "Stocks with MORE of `signal_field` earn higher returns", "examples": ["high-turnover stocks outperform"]},
        "less_outperforms": {"what": "Stocks with LESS of `signal_field` earn higher returns", "examples": ["illiquid, rarely traded stocks outperform"]}
      }
    },
    "horizon_stated": {
      "type": "noul",
      "instructions": "Does `specification` state a time period over which the signal is measured or held?"
    },
    "horizon": {
      "type": "choice",
      "instructions": "What time period over which the signal is measured does `specification` state?",
      "criteria": {
        "days": "A few days, up to about two weeks",
        "weeks_to_months": "About two weeks to three months",
        "months_to_year": "About three to twelve months, such as 'the past year'",
        "multi_year": "More than a year, such as 'three to five years'"
      }
    },
    "uses_volume": {
      "type": "noul",
      "instructions": {"question": "Does `specification` call for using `field`, or a direct equivalent of it?", "field": {"name": "volume", "meaning": "number of shares traded in the day"}}
    },
    "uses_shares": {
      "type": "noul",
      "instructions": {"question": "Does `specification` call for using `field`, or a direct equivalent of it?", "field": {"name": "shares", "meaning": "shares outstanding"}}
    },
    "uses_cap": {
      "type": "noul",
      "instructions": {"question": "Does `specification` call for using `field`, or a direct equivalent of it?", "field": {"name": "cap", "meaning": "market capitalization (price times shares outstanding)"}}
    },
    "uses_returns": {
      "type": "noul",
      "instructions": {"question": "Does `specification` call for using `field`, or a direct equivalent of it?", "field": {"name": "returns", "meaning": "the stock's daily return"}}
    },
    "uses_close": {
      "type": "noul",
      "instructions": {"question": "Does `specification` call for using `field`, or a direct equivalent of it?", "field": {"name": "close", "meaning": "daily closing price"}}
    }
  }
}
```

**Labels.** The 292 construction labels apply unchanged, because the
mutations only change the formula, and code does the comparison. Separately,
hand-label the 60 hypotheses' stated direction, horizon and fields. That shows
whether errors come from Jev's reading or from the code's formula analysis.

**Variants to run:**

- the same request with the two `direction` options in swapped order (the
  order effect);
- with and without `criteria` on the Nouls.

**Report:**

- AUROC per question;
- false passes and false rejections on the test half, at thresholds picked on
  the tuning half;
- the same counts for Claude Sonnet on the same split.

### Experiment 2: translate-first ablation (formula rendered in English)

This tests directly whether symbolic input caused the failure. Code renders
each formula as one deterministic English sentence that states the field,
the window in months and which end of the ranking it buys. Jev then compares
two plain-language statements, as in the
[citation-check](https://docs.typesafe.ai/cookbooks/citation_check) cookbook.
Run each item a second time with the two state fields in the opposite order,
and average the two answers.

```json
{
  "model": "jev-1.14.0",
  "state": {
    "hypothesis": "Stocks with persistently low share turnover over the past year earn higher subsequent returns than high-turnover stocks, consistent with a liquidity premium.",
    "signal": "Computes each stock's average daily share turnover (shares traded divided by shares outstanding) over the past 12 months, and buys the stocks with the HIGHEST turnover."
  },
  "questions": {
    "direction_relation": {
      "type": "choice",
      "instructions": "Does `signal` buy the stocks that `hypothesis` says should earn higher returns?",
      "criteria": {
        "same": "`signal` buys the stocks `hypothesis` expects to outperform",
        "opposite": "`signal` buys the stocks `hypothesis` expects to underperform",
        "unrelated": "`signal` ranks stocks on something `hypothesis` does not discuss"
      }
    },
    "same_quantity": {
      "type": "noul",
      "instructions": "Does `signal` measure the same quantity that `hypothesis` is about?"
    },
    "same_period": {
      "type": "noul",
      "instructions": "Does `signal` use the time period that `hypothesis` states?",
      "criteria": {
        "true": "The periods match roughly, or `hypothesis` states no period",
        "false": "`hypothesis` states a period and `signal` uses a clearly different one"
      }
    }
  }
}
```

The `same_period` criteria put the "no period stated" default into `true`. The
docs warn about default clauses like this. If it scores poorly, use the
companion-Noul split from Experiment 1. If Experiment 2 does no better than
the failed run, symbolic input was not the only problem. If it does well,
compare it with Experiment 1: the English rendering already contains the
direction that code computed.

### Experiment 3: paperlog relevance and tags

Send one request per paper: title plus abstract. Use the existing LLM
screenings as reference labels. They are LLM labels, not ground truth, so
hand-check a sample of the disagreements.

**Questions in each request:**

- two Scores (methods and findings), with relevance = the maximum, taken in
  code;
- one Noul per tag. All 18 are listed: they answer independently and add few
  tokens;
- one Choice for look-ahead bias.

```json
{
  "model": "jev-1.14.0",
  "state": {
    "title": "…",
    "abstract": "…"
  },
  "questions": {
    "methods_relevance": {
      "type": "score",
      "instructions": "How useful are the METHODS in `abstract` to a team that uses LLM agents to generate and statistically validate equity trading factors?",
      "criteria": [
        "Irrelevant",
        "Tangential",
        "Agent work only loosely applicable, or finance without a method we could use",
        "General agent methods or general multiple-testing / ML validation clearly transferable to factor research",
        "LLM or agentic finance research workflows, inference for strategy evaluation, or finance ML for return prediction",
        "LLM or agentic generation of trading signals, factors or economic hypotheses, or a validation method for many-candidate search we could adopt"
      ]
    },
    "findings_relevance": {
      "type": "score",
      "instructions": "How useful are the FINDINGS in `abstract` as a source of testable hypotheses about US stock returns from daily price, volume, return, share or market-cap data?",
      "criteria": [
        "Irrelevant",
        "Tangential",
        "Finance research without return-predictability content",
        "Return predictability in other markets or asset classes with a mechanism that could transfer to US equities",
        "Return predictability that needs data we may add (fundamentals, analyst forecasts), or evidence about anomalies as a group",
        "Documented predictability of US stock returns with a stated mechanism, testable with daily or monthly price, volume, return, shares or market-cap data"
      ]
    },
    "tag_hypothesis_generation": {"type": "noul", "instructions": "Is the paper about generating testable economic or trading hypotheses, by people or by LLMs?"},
    "tag_alpha_factor_mining": {"type": "noul", "instructions": "Is the paper about searching for or mining alpha factors or trading signals?"},
    "tag_agent_architecture": {"type": "noul", "instructions": "Is the paper about the design of LLM agent systems, such as tools, memory, planning or multi-agent setups?"},
    "tag_research_automation": {"type": "noul", "instructions": "Is the paper about automating scientific or financial research workflows?"},
    "tag_reproducibility_and_provenance": {"type": "noul", "instructions": "Is the paper about reproducibility, provenance or audit trails of research results?"},
    "tag_multiple_testing": {"type": "noul", "instructions": "Is the paper about multiple-testing corrections or false discovery control?"},
    "tag_backtest_overfitting": {"type": "noul", "instructions": "Is the paper about overfitting in backtests or the probability of backtest overfitting?"},
    "tag_sharpe_inference": {"type": "noul", "instructions": "Is the paper about statistical inference on Sharpe ratios, such as deflated or haircut Sharpe?"},
    "tag_cross_validation": {"type": "noul", "instructions": "Is the paper about cross-validation methods for time series or financial data?"},
    "tag_leakage_and_lookahead": {"type": "noul", "instructions": "Is the paper about data leakage or look-ahead bias?"},
    "tag_data_and_universe": {"type": "noul", "instructions": "Is the paper about financial datasets, data quality or how the stock universe is chosen?"},
    "tag_transaction_costs": {"type": "noul", "instructions": "Is the paper about transaction costs, trading frictions or capacity?"},
    "tag_portfolio_construction": {"type": "noul", "instructions": "Is the paper about building portfolios from signals, such as weighting or risk control?"},
    "tag_evaluation_benchmark": {"type": "noul", "instructions": "Does the paper introduce or use a benchmark for evaluating models or strategies?"},
    "tag_survey": {"type": "noul", "instructions": "Is the paper a survey or literature review?"},
    "tag_return_anomaly": {"type": "noul", "instructions": "Does the paper document a specific stock-return anomaly or predictable return pattern?"},
    "tag_factor_models": {"type": "noul", "instructions": "Is the paper about asset-pricing factor models, such as Fama-French or latent factors?"},
    "tag_finance_ml": {"type": "noul", "instructions": "Does the paper apply machine learning to financial prediction?"},
    "lookahead_bias": {
      "type": "choice",
      "instructions": "If the paper evaluates an LLM on historical market data, how does `abstract` handle leakage from the model's training data overlapping the test period?",
      "criteria": {
        "controlled": "Says how leakage is avoided or tested, such as test periods after the training cutoff, anonymised tickers or explicit leakage tests",
        "not_addressed": "Evaluates an LLM on historical market data and does not mention leakage",
        "not_applicable": "No LLM evaluation on historical market data",
        "unclear": "The abstract does not say enough to tell"
      }
    }
  }
}
```

**Report:**

- Spearman correlation between max(Score) and the LLM relevance score;
- per-tag AUROC against the LLM tags;
- per-tag thresholds tuned on half the papers and tested on the other half
  (expect high thresholds, because multi-label Nouls over-predict);
- agreement on look-ahead bias.

**Promotion rule.** Promote Jev only as a pre-filter (for example, skip the
LLM screen when both relevance Scores are confidently ≤ 1). Do this only if
the false-skip rate on relevance ≥ 4 papers is acceptable on the test half.

## Claims I could not verify

- **Second-hand claims in the review paper.** The review
  ([2609.32160](https://arxiv.org/abs/2609.32160)) summarizes 28 other papers.
  I read only the review's text for those, not the original papers. This
  covers the crash-coding recalibration (3.3-fold), the 0.01 probability grid,
  the context-steering flip rates, the mean confidence of wrong tool calls,
  the multi-step-arithmetic and derivation-checking weaknesses, and the
  "one Noul per option is better calibrated than a Choice" finding.
- **Vendor latency** ("about 100 ms" and "about 150ms") and the vendor cookbook
  numbers (self-consistency, SEC, citation check, parallel questions). These
  are the vendor's own measurements. The parallel-questions cost table is
  produced by code on the page, so I saw only the headline ratios.
- **Rate limits.** No numbers are published. They appear only in an
  account's console.
- **Formulas and code as input.** No official page and no paper I read tests
  Jev on structured formulas or code. The conclusion that it fits our task
  poorly is an inference from the documented weaknesses, plus our own run.
- **Version.** Every third-party result above used `jev-1.13.0`. TypeSafe's
  changelog lists no reasoning or symbolic improvements in 1.14
  ([changelog](https://docs.typesafe.ai/changelog)), but I found no
  independent evaluation of 1.14.
- **Unnamed papers.** The brief named eight arXiv papers. All eight are cited
  above. I did not look for other papers.

## Sources

TypeSafe documentation (fetched 2026-11-24):

- Introduction: <https://docs.typesafe.ai/introduction>
- System One: <https://docs.typesafe.ai/concepts/system-one>
- State: <https://docs.typesafe.ai/concepts/state>
- How to build with TypeSafe: <https://docs.typesafe.ai/concepts/how-to-build-with-system-one>
- Use cases: <https://docs.typesafe.ai/concepts/use-case-map>
- AI primer: <https://docs.typesafe.ai/introduction/machine-learning-primer>
- Primitives: <https://docs.typesafe.ai/primitives>
- Noul: <https://docs.typesafe.ai/primitives/noul>
- Advanced structure: <https://docs.typesafe.ai/primitives/advanced>
- Confidence: <https://docs.typesafe.ai/confidence>
- Model jaggedness: <https://docs.typesafe.ai/model-jaggedness>
- Models (price, context, rate limits, aliases): <https://docs.typesafe.ai/models>
- Model changelog: <https://docs.typesafe.ai/changelog>
- Request rules and errors: <https://docs.typesafe.ai/api/errors>
- API reference: <https://docs.typesafe.ai/api/system-one>
- OpenAPI spec: <https://api.typesafe.ai/openapi.json>
- Patterns: companion Noul <https://docs.typesafe.ai/patterns/companion-noul>; Noul screen <https://docs.typesafe.ai/patterns/noul-screen>
- Cookbooks: parallel questions <https://docs.typesafe.ai/cookbooks/parallel_questions>; self-consistency (Nouls) <https://docs.typesafe.ai/cookbooks/consistency_noul_cookbook>; verifying LLM answers <https://docs.typesafe.ai/cookbooks/verify_llm_outputs>; citation check <https://docs.typesafe.ai/cookbooks/citation_check>; classification using confidence <https://docs.typesafe.ai/cookbooks/classification_using_confidence>; hierarchical classification <https://docs.typesafe.ai/cookbooks/hierarchical_classification>
- Legal / data handling: <https://docs.typesafe.ai/legal>
- Full page index: <https://docs.typesafe.ai/llms.txt>

arXiv (abstract plus the HTML full text, v1):

- 2609.37647, Evaluating and Benchmarking the System One Model Jev: <https://arxiv.org/abs/2609.37647>
- 2609.32160, Typed Decision Models: An Early Evidence Audit and Evaluation Checklist: <https://arxiv.org/abs/2609.32160>
- 2610.06625, JEV versus LLMs: Accuracy, Cost and Calibration on Seven Political Science Replications: <https://arxiv.org/abs/2610.06625>
- 2610.11901, Can Decision Models Understand Stance? Evaluating Jev Against General-Purpose LLMs: <https://arxiv.org/abs/2610.11901>
- 2610.00376, A First Glance at Jev for Network Traffic Classification: <https://arxiv.org/abs/2610.00376>
- 2610.08829, Emo-Jev: Probabilistic Reasoning for Emotion Classification with Jev: <https://arxiv.org/abs/2610.08829>
- 2610.17205, Releasing the Inner Jev: A Simple Recipe for Open Decision Models: <https://arxiv.org/abs/2610.17205>
- 2610.02076, LLM-as-Jev: LLMs Are Already Jev-Style Decision Models: <https://arxiv.org/abs/2610.02076>
