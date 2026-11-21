#!/bin/sh
# Docker entrypoint (W5.5/W5.7 deploy contract).
#
# This wrapper deliberately does ONE thing: exec the Python launcher. It never
# `source`s or `eval`s the dotenv secret file — sourcing would execute its
# content as shell code (arbitrary command execution from a "config" file).
# All secret parsing, allowlisting and PROJECT_ENV validation happens in
# medcopilot/deploy/entrypoint.py (python-dotenv, data only), which then
# execs `uvicorn ... --workers 1`. See docs/PRODUCTION_DEPLOY.md §8.
#
# MED_RAG_SECRET_FILE overrides the docker-secrets default path (deployment
# choice; the launcher still parses it as data and still allowlists it).
#
# Interpreter hooks are scrubbed HERE, before the launcher's python3 starts:
# the launcher's own scrub (entrypoint.py) can only protect the process it
# execs, but an ambient PYTHONPATH is already in effect while the launcher
# imports dotenv and parses the secret file, and LD_PRELOAD cannot be removed
# from a running interpreter at all. Container cwd is /app (Dockerfile
# WORKDIR), so `-m` resolves the package without PYTHONPATH.
set -eu
unset PYTHONPATH LD_PRELOAD
exec python3 -m medcopilot.deploy.entrypoint "${MED_RAG_SECRET_FILE:-/run/secrets/med_env}"
