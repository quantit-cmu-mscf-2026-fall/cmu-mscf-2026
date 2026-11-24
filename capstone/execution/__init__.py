"""Execution: what may be traded and how it is named, and paper trading.

Contracts name what may be traded. The broker protocol, the paper-only guard
and a fake broker are paper only, dry-run by default: nothing here connects
anywhere unless the paper guard in `capstone.execution.broker` passes.
"""
