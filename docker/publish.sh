set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The base image installs everything from apt/pip and copies nothing in, so the
# build context is this directory rather than the whole repo.
docker build \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t mpalomata167/yolo:1.0.38 \
  "$SCRIPT_DIR"

docker push mpalomata167/yolo:1.0.38
