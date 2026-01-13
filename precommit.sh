#!/usr/bin/env bash

set -euo pipefail

uv run ruff check --no-fix ./src/ ./scripts/
uv run ruff format --check ./src/ ./scripts/
#uv run ty check ./src/ ./scripts/
