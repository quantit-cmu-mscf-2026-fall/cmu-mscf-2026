"""Report-only analyses of a run's search-period outputs (distributions, classes, decay).

Nothing here decides a candidate or opens the holdout: each analysis reads a
matrix and summary that already exist and writes data, `stats.json` and
numbered statements (`outputs.write_outputs`).
"""
