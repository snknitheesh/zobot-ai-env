#!/bin/bash
set -e
export DOCKER_BUILDKIT=1

# Ensure we are in the project root (simple check, or just assume)
# Using paths relative to project root, assuming script is run from there or we adjust context.
# Actually, let's make it work regardless of where it's called if we can, but "simple" is requested.
# I will assume the user runs it from the workspace root.

echo "Building base image (base_ai_image)..."
docker build -f .devcontainer/dockerfile.base_ai -t base_ai_image:latest .

# Build the new layer on top
echo "Building new layer (ai_new_image)..."
docker build -f .devcontainer/Dockerfile.ai_new -t ai_new_image:latest .

echo "All builds completed successfully!"
