# Zobot Lab - DevContainer-Based Robotics Development

GPU-accelerated development container with ROS 2 Jazzy, PyTorch, TensorFlow, TensorRT, VPI, Gazebo, and Conda.

## Prerequisites

- Docker with NVIDIA Container Toolkit
- Docker Compose v2
- NVIDIA GPU (RTX 5090 / Blackwell tested)

## First Time Setup

### 1. Add `zobot_lab` to your PATH

```bash
echo 'export PATH="/lake/workspaces/env_ws/.bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
zobot_lab --help
```

### 2. Get the images

**Option A: Pull base from GHCR + build overlay (recommended)**

```bash
zobot_lab --pull
```

This pulls the pre-built base image from `ghcr.io/snknitheesh/zobot-lab-jazzy-base:latest` and builds the lightweight overlay on top. Fastest way to get started.

**Option B: Build everything locally**

```bash
zobot_lab --build
```

Builds both base and overlay from source. Takes a long time (PyTorch, TensorFlow, ROS2, etc.).

To rebuild from scratch with no cache:

```bash
zobot_lab --rebuild
```

### 3. Start the container

```bash
zobot_lab
```

## Usage

| Command | Description |
|---------|-------------|
| `zobot_lab` | Start container and enter bash shell |
| `zobot_lab --pull` | Pull base from GHCR, build overlay |
| `zobot_lab --build` | Build base + overlay locally |
| `zobot_lab --rebuild` | Rebuild all from scratch (no cache) |
| `zobot_lab --conda myenv` | Start with a conda environment activated |
| `zobot_lab --vscode` | Open in VS Code (DevContainer) |
| `zobot_lab --prune` | Stop and remove containers (keeps images) |
| `zobot_lab --help` | Show help |

## What's Inside

### Base Image (`ghcr.io/snknitheesh/zobot-lab-jazzy-base`)

| Component | Version |
|-----------|---------|
| CUDA | 13.2 |
| Python | 3.12.12 (source build) |
| PyTorch | 2.11.0 (cu130) |
| Torchvision | 0.26.0 |
| TensorRT | 10.16.0 |
| TensorFlow | latest |
| VPI | 4.0 |
| ROS 2 | Jazzy |
| RMW | Zenoh |
| Gazebo | Harmonic |
| Conda | Miniconda |
| Pixi / UV | latest |
| PKL | 0.30.2 |

### Overlay Image (`zobot-lab-jazzy`)

- User setup (zozo, uid 1000)
- Conda init and configuration
- Pixi and UV for zozo user
- VCS repo import and colcon build
- Workspace aliases and scripts

## Directory Structure

```
env_ws/
├── .bin/                  # CLI tools (zobot_lab, colcon_runner, ros-mgr, req)
├── .dev/                  # Dev scripts (alias.sh, entrypoint, show_packages)
├── .devcontainer/
│   ├── Dockerfile.base    # Heavy base image (rarely rebuilt)
│   ├── Dockerfile.ai      # Lightweight overlay (fast rebuilds)
│   └── jazzy/
│       └── devcontainer.json
├── .requirements/
│   ├── additional_pip_requirements.txt
│   └── depend.repos       # VCS repos to import
├── debug/                 # Debug ROS2 packages (Python + C++)
├── docker-compose.yml
└── README.md
```

## Pushing Base Image to GHCR

```bash
# Login
gh auth token | docker login ghcr.io -u snknitheesh --password-stdin

# Tag and push
docker tag zobot-lab-jazzy-base:latest ghcr.io/snknitheesh/zobot-lab-jazzy-base:latest
docker push ghcr.io/snknitheesh/zobot-lab-jazzy-base:latest
```

## Mounted Volumes

| Host | Container | Purpose |
|------|-----------|---------|
| `.` (env_ws) | `/home/zozo/zobot_ws` | Workspace |
| `/lake/.conda_docker` | `/home/zozo/.conda` | Persistent conda envs |
| `~/.ssh` | `/home/zozo/.ssh` | SSH keys (read-only) |
| `~/.cache/huggingface` | `/home/zozo/.cache/huggingface` | HF tokens |
| `/tmp/.X11-unix` | `/tmp/.X11-unix` | X11 display |
| `/dev` | `/dev` | Device access |
