#!/bin/bash
# Script to ensure base image is built before DevContainer starts
set -e

echo "🔍 Checking if robotics-dev-base:latest exists..."

if docker image inspect robotics-dev-base:latest >/dev/null 2>&1; then
    echo "✅ Base image already exists"
else
    echo "🔨 Building base image..."
    docker compose --profile build build base-dev
    echo "✅ Base image built successfully"
fi
