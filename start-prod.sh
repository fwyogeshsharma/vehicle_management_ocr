#!/usr/bin/env bash
# Build and (re)deploy the OCR worker on the VM.   Usage: ./start-prod.sh [--no-pull]
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null || die "docker is not installed"

# Works with either the classic `docker-compose` (v1) or the `docker compose` plugin.
# If neither is there, install the classic one from the distro repo.
detect_compose() {
  if command -v docker-compose >/dev/null 2>&1; then DC="docker-compose"
  elif docker compose version >/dev/null 2>&1; then DC="docker compose"
  else return 1; fi
}
if ! detect_compose; then
  say "docker-compose not found; installing it with apt"
  apt-get update -y && apt-get install -y docker-compose || die "could not install docker-compose"
  detect_compose || die "docker-compose still not available"
fi
say "Using: $DC"

# Pull the latest code unless told not to.
if [[ "${1:-}" != "--no-pull" && -d .git ]]; then
  say "Pulling latest code"
  git pull --ff-only || die "git pull failed (use --no-pull to deploy what is on disk)"
fi

# Config
if [[ ! -f .env ]]; then
  [[ -f .env.example ]] || die ".env is missing and there is no .env.example"
  cp .env.example .env
  chmod 600 .env
  say "Created .env from .env.example -- check the values below"
fi
chmod 600 .env
set -a; source .env; set +a

for v in DB_USER DB_PASSWORD DB_NAME; do
  [[ -n "${!v:-}" ]] || die "$v is empty in .env"
done

if [[ "${VM_IMAGE_BACKEND:-local}" == "local" ]]; then
  [[ -d "${IMAGE_HOST_DIR:-}" ]] || die "IMAGE_HOST_DIR='${IMAGE_HOST_DIR:-}' is not a directory. Set it in .env to the folder the API saves photos into."
fi

# Is Postgres listening on this VM?
say "Checking PostgreSQL on 127.0.0.1:${DB_PORT:-5432}"
(exec 3<>"/dev/tcp/127.0.0.1/${DB_PORT:-5432}") 2>/dev/null \
  || die "nothing is listening on 127.0.0.1:${DB_PORT:-5432}"

# Build and start
say "Building image (first build takes several minutes)"
$DC build

say "Starting worker"
$DC down --remove-orphans >/dev/null 2>&1 || true
$DC up -d

# The worker checks DB + storage, loads models, then logs "polling". Wait for that.
say "Waiting for the worker to come up"
for _ in $(seq 1 60); do
  state=$(docker inspect -f '{{.State.Status}}' vehicle-ocr-worker 2>/dev/null || echo missing)
  if [[ "$state" != "running" ]]; then
    $DC logs --tail 40 ocr-worker
    die "worker is $state -- see the log above"
  fi
  if $DC logs ocr-worker 2>&1 | grep -q "polling every"; then
    say "Worker is up"
    $DC logs --tail 15 ocr-worker
    docker image prune -f >/dev/null
    printf '\nFollow logs:  %s logs -f ocr-worker\nStop:         %s down\n' "$DC" "$DC"
    exit 0
  fi
  sleep 2
done
$DC logs --tail 40 ocr-worker
die "worker did not start polling within 2 minutes"
