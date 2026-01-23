#!/bin/bash
# Script to install Qt dependencies for lerobot_ui

echo "Installing Qt dependencies for lerobot_ui..."

# Install xcb-cursor libraries
sudo apt-get update
sudo apt-get install -y libxcb-cursor0 libxcb-cursor-dev

# Install other Qt6 dependencies that might be needed
sudo apt-get install -y \
    libxcb-xinerama0 \
    libxcb-xinerama0-dev \
    libxcb-xfixes0 \
    libxcb-xfixes0-dev \
    libxcb-render0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shape0-dev \
    libxcb-keysyms1 \
    libxcb-keysyms1-dev \
    libxcb-image0 \
    libxcb-image0-dev \
    libxcb-icccm4 \
    libxcb-icccm4-dev \
    libxcb-sync1 \
    libxcb-sync-dev \
    libxcb-xkb1 \
    libxcb-xkb-dev \
    libxkbcommon-x11-0 \
    libxkbcommon-x11-dev \
    libxkbcommon0 \
    libxkbcommon-dev

echo "✅ Qt dependencies installed!"
echo ""
echo "Now try running: lerobot_ui"
