set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

docker build --no-cache \
  -f "${SCRIPT_DIR}/Dockerfile.deploy" \
  -t mpalomata167/yolo:1.0.38-deploy \
  "$PROJECT_ROOT"

docker push mpalomata167/yolo:1.0.38-deploy
