import os
from typing import Dict


def setup_environment(env_vars: Dict[str, str]) -> None:
    """Set up environment variables if not already set."""
    for var, value in env_vars.items():
        if not os.environ.get(var):
            os.environ[var] = value