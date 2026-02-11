#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Load .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Handle --reset
if [[ "${1:-}" == "--reset" ]]; then
    echo "Resetting user model, sessions, and plans..."
    rm -f data/user_model/model.json
    rm -f data/sessions/*
    rm -f data/plans/*
    echo "Done. Starting fresh onboarding."
    shift
fi

exec uv run python -m src.interface.cli "$@"
