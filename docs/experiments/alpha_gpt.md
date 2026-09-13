# Alpha-GPT Replication Experiment

Paper:
https://arxiv.org/html/2308.00016v2

## Goal

Perform a methodological replication of Alpha-GPT as one experiment
inside the existing capstone.

The central hypothesis to test is:

Does an LLM-guided
generate -> evaluate -> search -> interpret -> refine
workflow improve alpha candidates relative to one-shot LLM-generated seeds?

Exact numerical reproduction is not required because the paper relies on
proprietary data / alpha infrastructure.

## Experiment stages

### Seed

Trading idea
-> Quant Developer
-> N symbolic alpha expressions
-> deterministic evaluation

### Search Enhancement

Seed alphas
-> genetic search
-> deterministic evaluation

### Interaction + Search Enhancement

Best development candidate
-> Analyst
-> revised hypothesis
-> new seeds
-> genetic search
-> deterministic evaluation

Compare:

- Seed
- Search Enhancement
- Interaction + Search Enhancement

## Architecture rules

Claude is invoked only through:

    claude -p

Do not use the Anthropic Python API.

The LLM may:
- interpret trading ideas
- generate symbolic alpha expressions
- interpret supplied results
- propose refinements

The LLM may NOT:
- calculate IC
- calculate Sharpe
- invent performance
- manipulate raw market data directly
- generate arbitrary Python as an alpha
- decide statistical significance without deterministic results

Reuse existing project code wherever possible.

Do not duplicate:
- CRSP loaders
- strategy infrastructure
- validation logic
- trial counting
- deployment infrastructure

Every evaluated alpha variant must be recorded as a trial.

Keep TRAIN, VALIDATION, and TEST separate.

TRAIN:
search fitness

VALIDATION:
development and model selection

TEST:
final evaluation only
