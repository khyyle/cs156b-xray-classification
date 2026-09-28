"""
Machine-specific configuration loaded from a `.env` file at project root. 
see `.env.example`, and note every key it lists must be set
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

from radiology_cls.utils import PROJECT_ROOT

ENV_FILE = PROJECT_ROOT / ".env"
ENV_SCHEMA_FILE = PROJECT_ROOT / ".env.example"


def _load_required_settings() -> dict[str, str]:
    load_dotenv(ENV_FILE, override=True)
    required_names = list(dotenv_values(ENV_SCHEMA_FILE, interpolate=False))
    missing_names = [name for name in required_names if not os.environ.get(name)]
    if missing_names:
        raise RuntimeError(
            f"Missing required settings {', '.join(missing_names)} "
            f"set these in a root .env following the schema in .env.example)"
        )
    return {name: os.environ[name] for name in required_names}


_settings = _load_required_settings()

DATA_ROOT = Path(_settings["DATA_ROOT"])
TRAIN_CSV = Path(_settings["TRAIN_CSV"])
TEST_CSV = Path(_settings["TEST_CSV"])
SOLUTION_CSV = Path(_settings["SOLUTION_CSV"])

CLUSTER_ACCOUNT = _settings["CLUSTER_ACCOUNT"]
CLUSTER_GPU_PARTITION = _settings["CLUSTER_GPU_PARTITION"]
