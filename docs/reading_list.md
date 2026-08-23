# Reading list

Published work only — no internal material. Everything here is either freely
available or reachable through the CMU library.

You are not expected to read all of this, and you are certainly not expected to
read it before week 1. Start with the four marked **[start here]**; they cover
the shape of the problem. Pull the rest when your own project needs them.

Prof. Schafer may add or replace items from the Financial Data Science and ML
Capstone sequences — his additions take precedence over ours.

## The core problem: too many hypotheses

**[start here]** Harvey, Liu & Zhu, *…and the Cross-Section of Expected Returns*
(2016) — the paper that made the field admit its multiple-testing problem.
Read it for the framing, not the specific t-stat threshold.

Harvey & Liu, *Backtesting* (2015) — the deflated/haircut Sharpe ratio,
i.e. what your Sharpe is worth after accounting for how many you tried.

Bailey & López de Prado, *The Deflated Sharpe Ratio* (2014) — the version most
teams end up implementing.

Bailey, Borwein, López de Prado & Zhu, *The Probability of Backtest
Overfitting* (2015) — PBO, and the argument that backtest overfitting is the
default outcome rather than an accident.

## Multiple testing, properly

**[start here]** Benjamini & Hochberg, *Controlling the False Discovery Rate*
(1995) — the original FDR paper. Short, readable, and the method you will
probably use.

Benjamini & Yekutieli (2001) — FDR control under dependency, which is the case
you actually have (candidate signals are correlated with each other).

Storey, *A direct approach to false discovery rates* (2002) — q-values and
estimating the proportion of true nulls.

Romano & Wolf, *Stepwise multiple testing as formalized data snooping* (2005) —
the bootstrap-based alternative to FDR, used in the finance literature.

White, *A Reality Check for Data Snooping* (2000) — the earlier classic.

## Validation on time series

**[start here]** López de Prado, *Advances in Financial Machine Learning*
(2018), chapters 4–8 — purged and embargoed cross-validation, sample weights,
and why naive k-fold leaks. If you read one book chapter this semester, read
chapter 7.

Arlot & Celisse, *A survey of cross-validation procedures* (2010) — the general
statistical picture behind those chapters.

Cerqueira, Torgo & Mozetič, *Evaluating time series forecasting models* (2020) —
empirical comparison of validation schemes on real series.

## Agentic and automated research

**[start here]** Anthropic, *Building effective agents* (2024, engineering blog)
— plain description of when agentic patterns help and when a single call is
better. Also: *Claude Code best practices*.

Lu et al., *The AI Scientist* (Sakana AI, 2024) — end-to-end automated research
loop, with an honest account of what breaks.

Wang et al., *Voyager* (2023) — skill accumulation in an open-ended environment;
the "what does the agent remember about prior attempts?" question in a different
domain.

Yao et al., *ReAct* (2022) and Shinn et al., *Reflexion* (2023) — the reasoning/
acting loop and self-critique, both of which show up in every agent framework you
will touch.

Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet* (2023) —
read this next to Reflexion. The disagreement is the point, and it bears directly
on whether your agent can validate its own findings.

Zhang et al., *A Survey on Large Language Model Based Agents* (2023) — a map if
you want one.

The weekly paper round is where the current literature enters this project. These
are a floor, not a syllabus — bring what you find.

## Market microstructure and costs (when you get to execution)

Almgren & Chriss, *Optimal Execution of Portfolio Transactions* (2000).

Kyle, *Continuous Auctions and Insider Trading* (1985) — where price impact
comes from.

Frazzini, Israel & Moskowitz, *Trading Costs* (2018) — empirical costs at scale,
useful for sanity-checking whether a signal survives implementation.

## Statistical background, if a gap shows up

Efron & Hastie, *Computer Age Statistical Inference* (2016) — free PDF from the
authors. Chapters 15 and 20 cover large-scale testing and inference after
selection.

Ioannidis, *Why Most Published Research Findings Are False* (2005) — outside
finance, and the cleanest statement of the base-rate argument for why this whole
project exists.
