#!/bin/sh
set -e

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$$PATH"

# Wait for the Docker daemon to be available (up to 120 s).
max_wait=120
elapsed=0
while ! docker info >/dev/null 2>&1; do
    if [ "$$elapsed" -ge "$$max_wait" ]; then
        echo "error: Docker did not start within $${max_wait}s" >&2
        exit 1
    fi
    sleep 2
    elapsed=$$((elapsed + 2))
done

# Start Docker infrastructure (Neo4j, Qdrant, etc.)
{{FIELDNOTES_CMD}} up

# Hand off to the daemon (replaces this shell process).
exec {{FIELDNOTES_CMD}} serve --daemon
