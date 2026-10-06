# Validation decisions

Decisions that fix how candidate signals are validated. Each entry records what
was chosen and why, so later results can be read against it.

## V-21: factor data for incremental-value tests

**Decision.** Incremental-value tests regress candidate returns on the
**Fama-French 5 factors plus Momentum** (`Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`,
`RF`, `Mom`), taken from the **Ken French Data Library** at **daily** frequency,
matching the candidate-return pipeline.

**Loading.** Use `capstone.data.load_ff5_momentum()`. It loads the daily FF5
(2x3) file and the daily Momentum file and inner-joins them on date, so the
result covers only days present in both (FF5 starts 1963-07-01; Momentum starts
earlier). Values are decimal returns, and the library's `-99.99` / `-999`
missing-value codes are converted to `NaN`.

**Why `capstone/data.py`, not `shared_data.py`.** Ken French data is public and
freely redistributable, so it goes through the public-data loader in
`capstone/data.py`, which downloads from the library and caches locally.
`capstone.shared_data` is reserved for the licensed CRSP/Compustat pull from
WRDS that is shared only inside the team (see `docs/shared_data.md` and
`docs/data_sources.md`). Routing public factors through it would mix a public
source into the licensed path for no benefit.

**Not yet decided / not implemented.** The alpha regression itself (estimator,
standard errors, how `RF` is netted from candidate returns) is a separate item.
