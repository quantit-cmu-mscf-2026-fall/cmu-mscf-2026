"""`python -m capstone.execution --tag deploy-YYYY.MM.DD --as-of YYYY-MM-DD`: dry run only."""

import sys

from capstone.execution.cli import main

if __name__ == "__main__":
    sys.exit(main())
