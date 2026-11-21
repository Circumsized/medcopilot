# Deployment artifacts for the Med-RAG single-node production path (W5.5/W5.7).
#
# ``entrypoint.py`` is the container launcher (dotenv secret file -> allowlisted
# env -> exec uvicorn); the ``*.service``/``*.timer`` files are systemd units;
# ``docker-entrypoint.sh`` is a one-line exec wrapper around the launcher. See
# ``docs/PRODUCTION_DEPLOY.md`` for the operator manual.
