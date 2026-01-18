# Environment Workspace (env_ws) - Docker Infrastructure

**Last Updated:** 2025-12-25  
**Purpose:** Advanced Docker-based ROS2 development environment with persistent containers  
**Owner:** zozo (snknitheesh@gmail.com)

---

## Overview

The `env_ws` workspace implements a sophisticated Docker containerization system for ROS2 robotics development. It features:

- **Multi-stage Docker builds** with layer caching
- **Persistent container management** (survive reboots, no data loss)
- **Dual ROS2 distro support** (Humble & Jazzy)
- **Integrated Conda** environment management
- **GPU support** (NVIDIA with full capabilities)
- **Advanced build system** (custom colcon wrapper)
- **VCS repository management** for external ROS packages
- **Hot-reload development** (workspace mounted, no rebuild needed)

### Architecture Philosophy

This system separates **build-time** (Docker image) from **runtime** (persistent container), enabling:
1. Fast iteration cycles (no image rebuilds for code changes)
2. Persistent conda environments across sessions
3. Multi-project workspace organization
4. Seamless host-container integration

---

## Directory Structure

```
env_ws/
├── .cursor/                         # Cursor IDE configuration
│   └── README.md                    # This file
├── .devcontainer/                   # Docker & devcontainer config
│   ├── Dockerfile.ai                # Multi-stage Dockerfile (199 lines)
│   └── devcontainer.json            # VS Code devcontainer config
├── .bin/                            # Container management scripts
│   ├── zobot_lab                    # Main container launcher (309 lines)
│   ├── colcon_runner.py             # Custom colcon wrapper (313 lines)
│   ├── ros-mgr                      # VCS repository manager (219 lines)
│   ├── req                          # Requirements updater (101 lines)
│   ├── alias.sh                     # Bash aliases (59 lines)
│   ├── ubuntu_bashrc                # Container bashrc (203 lines)
│   └── colcon_defaults.yaml         # Colcon configuration
├── .requirements/                   # Build-time dependencies
│   ├── additional_ros_requirements.txt
│   ├── additional_pip_requirements.txt
│   ├── additional_deb_requirements.txt
│   └── depend.repos                 # VCS repositories (YAML format)
├── ros/                             # ROS2 packages (mounted at runtime)
├── workspaces/                      # Empty placeholder
├── config.yaml                      # Camera monitoring config
├── test.py                          # Camera performance monitor
└── performance.csv                  # Performance data log
```

---

## Docker Architecture

### Multi-Stage Build System

The Dockerfile implements **7 build stages** with progressive complexity:

```
BASE_IMAGE (rwthika/ros2-ml:humble/jazzy)
    ↓
├── underlay-builder      # User setup, system packages
    ↓
├── conda-builder         # Conda installation & configuration
    ↓
├── requirements-builder  # Python/ROS/APT dependencies
    ↓
├── sim-builder          # Gazebo Harmonic simulation
    ↓
└── overlay-builder      # Final stage: workspace setup
```

### Stage 1: Underlay Builder

**Purpose:** Base system setup

```dockerfile
FROM ${BASE_IMAGE} AS underlay-builder
```

**Actions:**
- Create user `zozo` (UID 1000, GID 1000)
- Grant passwordless sudo access
- Install essential build tools (gcc, cmake, git, vim, etc.)
- Configure GPU environment variables

**Key Features:**
- Non-root user execution
- Sudo privileges for runtime package installation
- Docker cache mounting for faster APT operations

### Stage 2: Conda Builder

**Purpose:** Python environment management

```dockerfile
FROM underlay-builder AS conda-builder
```

**Actions:**
- Download Miniconda3 (latest Linux x86_64)
- Install to `/home/zozo/.anaconda`
- Configure conda (disable auto-activation, accept TOS)
- Set system-wide conda configuration

**Key Features:**
- User-owned conda installation
- Configured for manual environment activation
- Cache-mounted download for rebuild speed
- System-wide conda config in `/etc/environment`

### Stage 3: Requirements Builder

**Purpose:** Install all dependencies

```dockerfile
FROM conda-builder AS requirements-builder
```

**Dependency Installation Order:**
1. **ROS packages** (`additional_ros_requirements.txt`)
   - python3-vcstool, python3-colcon-common-extensions, python3-rosdep
2. **System packages** (`additional_deb_requirements.txt`)
   - speech-dispatcher
3. **PyTorch** (reinstalled from CUDA 12.8 nightly builds)
   - Uninstall CPU versions
   - Install GPU versions: `torch`, `torchvision`, `torchaudio`
4. **Python packages** (`additional_pip_requirements.txt`)
   - catkin_pkg, uv, pyrealsense2
5. **Build tools** (upgraded)
   - pip, setuptools, colcon-core

**Cache Strategy:**
- APT cache: `/var/cache/apt`
- Pip cache: `/root/.cache/pip`
- Significantly speeds up rebuilds

### Stage 4: Sim Builder

**Purpose:** Gazebo simulation integration

```dockerfile
FROM requirements-builder AS sim-builder
```

**Actions:**
- Add OSRF GPG key and repository
- Install Gazebo Harmonic (`gz-harmonic`)

**Use Case:**
- Robot simulation in containerized environment
- Physics-based testing without hardware

### Stage 5: Overlay Builder (Final Stage)

**Purpose:** Workspace configuration and VCS integration

```dockerfile
FROM sim-builder AS overlay-builder
```

**Directory Setup:**
- `$ZOBOT_WS` → `/home/zozo/zobot_ws` (workspace root, mounted from host)
- `/home/zozo/.ros_builds` → Build artifacts (build/, install/, log/)
- `/home/zozo/mount` → Optional external mounts
- `/home/zozo/cave` → Shared storage mount (`/mnt/cave` on host)
- `/opt/ros/ros_ws` → VCS overlay workspace

**Configuration Files Copied:**
- `.bashrc` → Enhanced shell with ROS auto-sourcing, conda init, aliases
- `alias.sh` → Custom aliases (cr, req, ros-mgr, Docker shortcuts)
- `colcon_runner.py` → Build system wrapper
- `colcon_defaults.yaml` → Default colcon configuration
- `ros-mgr` → VCS repository manager
- `req` → Requirements updater

**VCS Repository Handling:**
1. Copy `depend.repos` to `/tmp/`
2. Import repositories to `/opt/ros/ros_ws` using `vcs import`
3. Build all imported packages with colcon
4. Install to `/home/zozo/.ros_builds/install`

**Environment Variables:**
- `ROS_DISTRO` → humble/jazzy (build arg)
- `RCUTILS_CONSOLE_OUTPUT_FORMAT` → Simplified logging
- `RCUTILS_COLORIZED_OUTPUT=1` → Colored logs
- `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` → DDS implementation
- `PYTHONDONTWRITEBYTECODE=1` → No .pyc files
- `CARB_AUDIO_DISABLE=1` → Disable CARB audio warnings
- `MAX_JOBS=8` → Parallel build jobs

---

## Container Management System

### Primary Tool: `zobot_lab`

**Location:** `.bin/zobot_lab`  
**Purpose:** Unified container lifecycle management

#### Commands

```bash
# Start container with ROS Humble (default)
zobot_lab

# Start with ROS Jazzy
zobot_lab --jazzy

# Activate conda environment on start
zobot_lab --conda robotics

# Mount external volume
zobot_lab --volume /path/to/data

# Build image only (no container start)
zobot_lab --build

# Remove container and image
zobot_lab --prune

# Show help
zobot_lab --help
```

#### Container Naming Convention

- **Image:** `zobot-lab-{distro}` (e.g., `zobot-lab-humble`)
- **Container:** `zobot-lab-{distro}` (e.g., `zobot-lab-humble`)
- **Hostname:** `zobot-{distro}` (e.g., `zobot-humble`)

#### Persistent Container Strategy

**Key Insight:** Containers run as **background daemons** (`tail -f /dev/null`)

**Lifecycle:**
1. **First run:** `docker run -d` creates persistent container
2. **Subsequent runs:** `docker exec -it` enters existing container
3. **Prune:** `docker rm` removes container, `docker rmi` removes image

**Benefits:**
- Data persists between sessions
- Conda environments survive
- No rebuild needed for code changes
- Fast startup after first launch

#### Volume Mounts

**Automatic Mounts:**
```bash
/tmp/.X11-unix → X11 display forwarding
$HOME/.ssh → SSH keys (read-only)
/mnt/cave → Shared storage
$WORKSPACE_ROOT → /home/zozo/zobot_ws
/dev → Device access (USB, serial, etc.)
/run/udev → Udev rules
/etc/group, /etc/passwd → User/group mapping
```

**Conditional Mounts:**
```bash
# Conda (if /mnt/cave/.conda_docker exists)
/mnt/cave/.conda_docker → /home/zozo/.anaconda

# Wandb API key (if ~/.netrc exists)
$HOME/.netrc → /home/zozo/.netrc

# Hugging Face tokens (if ~/.cache/huggingface exists)
$HOME/.cache/huggingface → /home/zozo/.cache/huggingface

# Custom volume (if --volume specified)
<custom_path> → /home/zozo/zobot_ws/mount
```

#### GPU Configuration

**Docker Args:**
```bash
--gpus all              # All GPUs accessible
--runtime nvidia        # NVIDIA container runtime
--privileged            # Full device access
```

**Environment:**
```bash
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=all
```

**Result:** Full CUDA, cuDNN, TensorRT access inside container

#### Network Configuration

```bash
--network host          # Share host network stack
--env ROS_DOMAIN_ID     # Inherit host's ROS_DOMAIN_ID
```

**Benefits:**
- No port mapping needed
- ROS2 DDS works transparently
- rviz2/gazebo GUI accessible from host

---

## Build System: `colcon_runner` (cr)

**Location:** `.bin/colcon_runner.py`  
**Purpose:** Simplified colcon interface with smart defaults

### Features

1. **Automatic Workspace Detection**
   - Searches for `ros/` folder in current or parent directory
   - Falls back to `/home/zozo/zobot_ws`

2. **Centralized Build Directory**
   - Build artifacts: `/home/zozo/.ros_builds/`
   - Keeps workspace clean
   - Shared across all projects

3. **Smart Package Selection**
   - Build specific packages or all packages
   - Validates package names before build
   - Lists available packages

4. **Optimized Defaults**
   - Symlink install (no copy overhead)
   - Release build type
   - Testing disabled by default
   - 8 parallel jobs

### Commands

```bash
# Build all packages
cr ba            # or: cr b

# Build specific package(s)
cr b my_package
cr b pkg1 pkg2 pkg3

# Build from external folder
cr bf /opt/ros/ros_ws

# Test all packages
cr ta            # or: cr t

# Test specific package(s)
cr t my_package

# Clean all build artifacts
cr ca            # or: cr c

# List available packages
cr p

# Show help
cr h
```

### Build Process

1. **Source ROS:** `/opt/ros/${ROS_DISTRO}/setup.bash`
2. **Run Colcon:**
   ```bash
   colcon build \
       --symlink-install \
       --cmake-args -DCMAKE_BUILD_TYPE=Release \
       --base-paths /home/zozo/zobot_ws/ros \
       --build-base /home/zozo/.ros_builds/build \
       --install-base /home/zozo/.ros_builds/install
   ```
3. **Source Workspace:** `source /home/zozo/.ros_builds/install/setup.bash`

### Configuration

**File:** `.bin/colcon_defaults.yaml`

```yaml
build:
  symlink-install: true
  cmake-args:
    - "-DCMAKE_BUILD_TYPE=Release"
    - "-DBUILD_TESTING=OFF"
  base-paths:
    - "/home/zozo/zobot_ws/ros"
  build-base: "/home/zozo/.ros_builds/build"
  install-base: "/home/zozo/.ros_builds/install"
  log-base: "/home/zozo/.ros_builds/log"
```

---

## VCS Repository Manager: `ros-mgr`

**Location:** `.bin/ros-mgr`  
**Purpose:** Manage external ROS packages via VCS (Version Control System)

### Repository Configuration

**File:** `.requirements/depend.repos` (YAML format)

**Example:**
```yaml
repositories:
  realsense_ros:
    type: git
    url: https://github.com/IntelRealSense/realsense-ros.git
    version: jazzy
  isaac_ros_common:
    type: git
    url: https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
    version: main
```

### Commands

```bash
# Import repositories to /opt/ros/ros_ws
ros-mgr import

# Check repository status
ros-mgr status

# Pull latest changes
ros-mgr pull

# Build imported repositories
ros-mgr build

# Install rosdep dependencies
ros-mgr install

# Clean repositories and build artifacts
ros-mgr clean

# Show help
ros-mgr help
```

### Workflow

1. **Edit** `depend.repos` with repository definitions
2. **Import:** `ros-mgr import` → Clone to `/opt/ros/ros_ws`
3. **Install Deps:** `ros-mgr install` → Run rosdep
4. **Build:** `ros-mgr build` → Colcon build
5. **Source:** `source ~/.ros_builds/install/setup.bash`

### Integration with Docker Build

During image build, `depend.repos` is:
1. Copied to `/tmp/depend.repos`
2. Imported to `/opt/ros/ros_ws` (if non-empty)
3. Built automatically
4. Installed to `/home/zozo/.ros_builds/install`

**Result:** External packages available immediately in new containers

---

## Requirements Manager: `req`

**Location:** `.bin/req`  
**Purpose:** Update dependencies inside running container

### Commands

```bash
# Update all requirements
req update

# Show help
req help
```

### Update Process

**Executed in order:**

1. **VCS Repositories** (`depend.repos`)
   - Import via `ros-mgr import`
   - Build via `ros-mgr build`

2. **ROS Packages** (`additional_ros_requirements.txt`)
   - Update APT cache
   - Install packages via `apt-get install`

3. **System Packages** (`additional_deb_requirements.txt`)
   - Install Debian packages

4. **Python Packages** (`additional_pip_requirements.txt`)
   - Upgrade pip
   - Install packages with `--upgrade --force-reinstall`

5. **Cleanup**
   - `apt-get autoremove`
   - `apt-get autoclean`

### Use Cases

- Add new dependencies without rebuilding image
- Update packages to latest versions
- Hot-fix missing dependencies during development

---

## Bash Environment

**File:** `.bin/ubuntu_bashrc` (203 lines)

### Features

1. **Enhanced Prompt**
   - Cyan username, green hostname, blue path
   - Robot emoji in terminal title: 🤖 ZoBot AI

2. **Color Support**
   - Colored `ls`, `grep`, `fgrep`, `egrep`
   - Syntax highlighting for common tools

3. **Git Aliases**
   - `gs` → `git status`
   - `ga` → `git add`
   - `gc` → `git commit`
   - `gp` → `git push`
   - `gl` → `git pull`
   - `gd` → `git diff`
   - `gb` → `git branch`
   - `gco` → `git checkout`

4. **Navigation Aliases**
   - `ws` → `cd /home/zozo/zobot_ws`
   - `rosws` → `cd /home/zozo/zobot_ws/ros`
   - `logs` → `cd /home/zozo/.ros_builds/log`
   - `builds` → `cd /home/zozo/.ros_builds`
   - `cave` → `cd /home/zozo/cave`

5. **Docker Aliases**
   - `dps` → `docker ps`
   - `dpa` → `docker ps -a`
   - `di` → `docker images`
   - `dstop` → `docker stop`
   - `drm` → `docker rm`
   - `drmi` → `docker rmi`

6. **ROS Aliases**
   - `rostopic` → `ros2 topic`
   - `rosnode` → `ros2 node`
   - `rosrun` → `ros2 run`
   - `roslaunch` → `ros2 launch`

7. **System Aliases**
   - `gpu` → `nvidia-smi`
   - `ports` → `netstat -tulpn | grep LISTEN`
   - `usage` → `du -h --max-depth=1 | sort -hr`
   - `meminfo` → `free -h`

8. **Colcon Shortcuts**
   - `crb` → `cr b` (build all)
   - `crt` → `cr t` (test all)
   - `crc` → `cr c` (clean all)
   - `crp` → `cr p` (list packages)

9. **Conda Integration**
   - Auto-initialize conda on shell start
   - Disable base environment auto-activation
   - `activate_conda` function for quick activation
   - Alias: `ca <env>` → Activate conda environment

10. **ROS Auto-Sourcing**
    - Sources `/opt/ros/${ROS_DISTRO}/setup.bash`
    - Sources `/home/zozo/.ros_builds/install/setup.bash`
    - Automatic on shell start

11. **Welcome Message**
    ```
    🤖 Welcome to ZoBot AI Container
    ROS Distribution: jazzy
    ROS Domain ID: 0
    Type 'cr h' for colcon commands or 'req help' for requirements management
    Quick navigation: rosws, ws, cdmount, cdcave
    ```

### Environment Variables

```bash
# Colors for scripts
GREEN, CYAN, YELLOW, RED, BLUE, PURPLE, NC

# PATH additions
PATH=/home/zozo/zobot_ws/.bin:/home/zozo/.anaconda/bin:$PATH

# Python
PYTHONDONTWRITEBYTECODE=1

# ROS
ROS_DISTRO=jazzy/humble
ROS_DOMAIN_ID=0 (inherited from host)
RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}]: {message}"
RCUTILS_COLORIZED_OUTPUT=1
RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# CUDA
PATH=/usr/local/cuda/bin:$PATH
LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Build
MAX_JOBS=$(nproc)

# Misc
CARB_AUDIO_DISABLE=1
CONDA_AUTO_ACTIVATE_BASE=false
```

---

## VS Code DevContainer Integration

**File:** `.devcontainer/devcontainer.json`

### Configuration

```json
{
  "name": "zobot-lab",
  "remoteUser": "zozo",
  "build": {
    "dockerfile": "Dockerfile.ai",
    "context": "..",
    "args": {"ROS_DISTRO": "jazzy"},
    "target": "overlay-builder"
  }
}
```

### Features

1. **DevContainer Features**
   - Common utilities
   - Pre-commit hooks
   - Docker-in-Docker (nested containers)
   - User configuration

2. **VS Code Extensions**
   - Python (ms-python.python)
   - Ruff linter (charliermarsh.ruff)
   - C/C++ tools (ms-vscode.cpptools-extension-pack)
   - ROS support (ms-iot.vscode-ros)
   - URDF viewer (smilerobotics.urdf)
   - XML/TOML/Markdown support

3. **File Associations**
   - `.rviz` → YAML
   - `.srdf, .urdf, .xacro` → XML

4. **Python Configuration**
   - Interpreter: `/home/zozo/.anaconda/bin/python`
   - Shell: `/bin/bash`

5. **Mounts**
   - Workspace: `$PWD → /home/zozo/zobot_ws`
   - Cave: `/mnt/cave → /home/zozo/cave`
   - SSH: `~/.ssh → /home/zozo/.ssh` (read-only)

6. **Runtime Args**
   - `--net=host` → Host networking
   - `--gpus=all` → GPU access
   - `--runtime=nvidia` → NVIDIA runtime
   - `--privileged` → Full device access
   - `--hostname=🤖zobot-jazzy` → Cute hostname with emoji

### Usage

**Open in VS Code:**
```bash
cd /home/zozo/workspaces/env_ws
code .
# Click "Reopen in Container" when prompted
```

**Command Palette:**
- `Dev Containers: Rebuild Container` → Rebuild image
- `Dev Containers: Reopen in Container` → Start/enter container
- `Dev Containers: Reopen Locally` → Exit to host

---

## Advanced Features

### Conda Persistence

**Problem:** Conda environments lost on container removal  
**Solution:** Mount `/mnt/cave/.conda_docker` to `/home/zozo/.anaconda`

**Setup:**
1. Create conda environment inside container
2. On `zobot_lab --prune`, conda dir is copied to `/mnt/cave/.conda_docker`
3. On next container start, conda dir is mounted back
4. Conda environments persist across container lifecycles

### API Key Persistence

**WandB:**
- File: `~/.netrc`
- Mount: `$HOME/.netrc → /home/zozo/.netrc`
- Login persists across sessions

**Hugging Face:**
- File: `~/.cache/huggingface/stored_tokens`
- Mount: `$HOME/.cache/huggingface → /home/zozo/.cache/huggingface`
- Tokens persist across sessions

### Multi-Distro Support

**Switch Between Distros:**
```bash
# Humble container
zobot_lab --humble

# Jazzy container
zobot_lab --jazzy
```

**Isolation:**
- Separate images: `zobot-lab-humble`, `zobot-lab-jazzy`
- Separate containers: `zobot-lab-humble`, `zobot-lab-jazzy`
- Can run simultaneously (different ROS_DOMAIN_ID)

### Device Access

**USB Devices:**
- `/dev` mounted → All USB devices accessible
- `/run/udev` mounted → Udev rules available
- User in `dialout` group → Serial port access

**Examples:**
- RealSense cameras: `/dev/video*`
- Serial devices: `/dev/ttyUSB*`, `/dev/ttyACM*`
- USB devices: `/dev/bus/usb/*`

---

## Workflow Examples

### Example 1: New ROS2 Package Development

```bash
# 1. Start container
zobot_lab --jazzy

# 2. Create new package
cd ~/zobot_ws/ros
ros2 pkg create my_package --build-type ament_python

# 3. Write code in VS Code (mounted volume, changes visible in container)
code ~/zobot_ws/ros/my_package

# 4. Build package
cr b my_package

# 5. Source and run
source ~/.ros_builds/install/setup.bash
ros2 run my_package my_node

# 6. Exit container (data persists)
exit

# 7. Re-enter later (package still built)
zobot_lab --jazzy
ros2 run my_package my_node  # Works immediately
```

### Example 2: Adding External ROS Package

```bash
# 1. Edit depend.repos
nano ~/zobot_ws/.requirements/depend.repos

# Add:
# repositories:
#   realsense_ros:
#     type: git
#     url: https://github.com/IntelRealSense/realsense-ros.git
#     version: jazzy

# 2. Start container
zobot_lab --jazzy

# 3. Import and build
ros-mgr import
ros-mgr install
ros-mgr build

# 4. Source and use
source ~/.ros_builds/install/setup.bash
ros2 pkg list | grep realsense
```

### Example 3: Conda Environment Workflow

```bash
# 1. Start container with conda env
zobot_lab --jazzy --conda robotics

# 2. Inside container, conda env is active
(robotics) zozo@zobot-jazzy:~/zobot_ws$

# 3. Install packages in conda
conda install pytorch torchvision -c pytorch

# 4. Run Python code with conda packages
python train_model.py

# 5. Prune container (conda saved to /mnt/cave/.conda_docker)
exit
zobot_lab --prune --jazzy

# 6. Restart with conda (environment restored)
zobot_lab --jazzy --conda robotics
# Conda packages still available!
```

### Example 4: GPU-Accelerated Training

```bash
# 1. Start container (GPU auto-configured)
zobot_lab --jazzy

# 2. Verify GPU access
nvidia-smi  # Shows GPU info

# 3. Run PyTorch training
python3 -c "import torch; print(torch.cuda.is_available())"  # True
python3 train_neural_network.py  # Uses GPU

# 4. Run ROS2 node with GPU inference
ros2 run my_package gpu_inference_node
```

### Example 5: Multi-Project Setup

```bash
# Project 1: Perception
cd ~/workspaces/env_ws
zobot_lab --jazzy
# Develop perception pipelines

# Project 2: Navigation (simultaneously)
# Open new terminal
cd ~/workspaces/nav_ws
zobot_lab --jazzy  # Enters same container
# Develop navigation stack

# Both projects share:
# - Same ROS2 installation
# - Same build artifacts
# - Same conda environments
# - Same GPU access
```

---

## Troubleshooting

### Issue: Container Won't Start

**Symptoms:** `zobot_lab` hangs or errors

**Solutions:**
```bash
# 1. Check Docker daemon
sudo systemctl status docker

# 2. Check for port conflicts
sudo netstat -tulpn | grep LISTEN

# 3. Remove stale containers
docker ps -a | grep zobot
docker rm -f zobot-lab-jazzy

# 4. Rebuild image
zobot_lab --build --jazzy
```

### Issue: GPU Not Available

**Symptoms:** `nvidia-smi` fails, `torch.cuda.is_available()` returns False

**Solutions:**
```bash
# 1. Check NVIDIA driver on host
nvidia-smi

# 2. Check nvidia-docker runtime
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# 3. Verify container runtime
docker inspect zobot-lab-jazzy | grep -i runtime

# 4. Restart Docker with NVIDIA runtime
sudo systemctl restart docker
```

### Issue: ROS2 Topics Not Visible

**Symptoms:** `ros2 topic list` shows no topics from host/other containers

**Solutions:**
```bash
# 1. Check ROS_DOMAIN_ID matches
# Inside container:
echo $ROS_DOMAIN_ID
# On host:
echo $ROS_DOMAIN_ID

# 2. Check network mode
docker inspect zobot-lab-jazzy | grep NetworkMode
# Should be "host"

# 3. Check FastRTPS configuration
echo $RMW_IMPLEMENTATION
# Should be "rmw_fastrtps_cpp"

# 4. Restart container with correct ROS_DOMAIN_ID
export ROS_DOMAIN_ID=42
zobot_lab --jazzy
```

### Issue: Conda Environment Not Found

**Symptoms:** `activate_conda myenv` fails

**Solutions:**
```bash
# 1. Check conda initialization
which conda
conda --version

# 2. List available environments
conda env list

# 3. Create environment if missing
conda create -n myenv python=3.10

# 4. Check conda mount
docker inspect zobot-lab-jazzy | grep anaconda
# Should show mount from /mnt/cave/.conda_docker
```

### Issue: Build Fails with Permission Errors

**Symptoms:** Colcon build fails with "Permission denied"

**Solutions:**
```bash
# 1. Check file ownership in workspace
ls -la ~/zobot_ws/ros/

# 2. Fix ownership if needed
sudo chown -R zozo:zozo ~/zobot_ws/

# 3. Check build directory permissions
ls -ld ~/.ros_builds/
chmod 755 ~/.ros_builds/

# 4. Clean and rebuild
cr ca
cr ba
```

### Issue: Out of Disk Space

**Symptoms:** Docker build fails, container creation fails

**Solutions:**
```bash
# 1. Check disk usage
df -h
docker system df

# 2. Remove unused Docker resources
docker system prune -a

# 3. Clean build artifacts
cr ca

# 4. Remove old containers/images
docker ps -a  # List all containers
docker rm $(docker ps -aq)  # Remove all stopped
docker images  # List images
docker rmi <image_id>  # Remove specific image
```

---

## Performance Optimization

### Build Speed

1. **Use BuildKit:**
   ```bash
   export DOCKER_BUILDKIT=1
   zobot_lab --build --jazzy
   ```

2. **Cache Mounts:**
   - APT cache: Already configured in Dockerfile
   - Pip cache: Already configured in Dockerfile

3. **Parallel Builds:**
   ```bash
   export MAX_JOBS=16
   cr ba
   ```

4. **Ccache for C++:**
   ```bash
   # Inside container
   sudo apt-get install ccache
   export PATH=/usr/lib/ccache:$PATH
   export CCACHE_DIR=/home/zozo/.ccache
   ```

### Runtime Performance

1. **Host Networking:**
   - Already enabled (`--net=host`)
   - No bridge overhead

2. **GPU Direct Access:**
   - Already enabled (`--gpus all`, `--runtime nvidia`)
   - No virtualization overhead

3. **SSD Storage:**
   - Store `/home/zozo/.ros_builds/` on SSD
   - Faster colcon builds

4. **RAM Disk for Logs:**
   ```bash
   # Mount tmpfs for log directory
   docker run --tmpfs /home/zozo/.ros_builds/log:rw,size=1G
   ```

---

## Security Considerations

### User Privileges

- **Non-root user:** All operations as `zozo` (UID 1000)
- **Sudo access:** Passwordless (convenience for development)
- **Production:** Remove sudo or require password

### SSH Keys

- **Mounted read-only:** `~/.ssh → /home/zozo/.ssh:ro`
- **Git operations:** SSH agent forwarding recommended
- **Never commit keys:** Already `.gitignore`d

### API Keys

- **Mounted dynamically:** Only if files exist on host
- **Read-write:** Allows updates inside container
- **Environment vars:** Alternative to file-based storage

### Device Access

- **Privileged mode:** Required for `/dev` access
- **Security risk:** Full host device access
- **Mitigation:** Run only on trusted networks

---

## Comparison with Alternatives

### vs. Native ROS2 Installation

| Feature | Docker (env_ws) | Native |
|---------|-----------------|--------|
| **Isolation** | ✅ Perfect | ❌ Shared |
| **Multi-distro** | ✅ Easy | ❌ Difficult |
| **Reproducibility** | ✅ Dockerfile | ❌ Manual |
| **Cleanup** | ✅ `docker prune` | ❌ Manual uninstall |
| **Performance** | ✅ Near-native | ✅ Native |
| **GPU Access** | ✅ Full | ✅ Full |
| **Setup Time** | ⚠️ 15-30 min build | ✅ 5-10 min install |

### vs. Docker Compose

| Feature | zobot_lab | docker-compose |
|---------|-----------|----------------|
| **Ease of Use** | ✅ Simple CLI | ⚠️ YAML config |
| **Persistence** | ✅ Built-in | ⚠️ Manual volumes |
| **Multi-distro** | ✅ Flag-based | ❌ Multiple files |
| **Conda Integration** | ✅ Automatic | ❌ Manual |
| **Interactive** | ✅ Direct exec | ⚠️ Via `docker-compose exec` |

### vs. VS Code DevContainer

| Feature | zobot_lab | DevContainer |
|---------|-----------|--------------|
| **IDE Integration** | ❌ CLI only | ✅ VS Code native |
| **Flexibility** | ✅ Any terminal | ⚠️ VS Code only |
| **Conda Support** | ✅ Flag-based | ⚠️ Manual config |
| **Speed** | ✅ Instant re-entry | ⚠️ VS Code restart |
| **Portability** | ✅ Any machine | ⚠️ VS Code required |

**Recommendation:** Use `zobot_lab` for terminal workflows, DevContainer for VS Code development.

---

## Future Enhancements

### Planned Features

- [ ] **Multi-user support** (different UIDs)
- [ ] **Kubernetes deployment** (container orchestration)
- [ ] **Remote development** (SSH into container from remote)
- [ ] **Integrated GUI apps** (VNC/X11 forwarding improvements)
- [ ] **Resource limits** (CPU/memory quotas)
- [ ] **Health checks** (automated container monitoring)
- [ ] **Backup/restore** (conda environments, configs)
- [ ] **Cache layers** (pre-built common packages)

### Experimental Features

- **Podman support** (Docker alternative)
- **ARM64 builds** (Raspberry Pi, Jetson)
- **ROS1 bridge** (ROS1 <-> ROS2 communication)
- **Isaac Sim integration** (NVIDIA Omniverse)

---

## Quick Reference

### Essential Commands

```bash
# Container Management
zobot_lab                      # Start/enter container (Humble)
zobot_lab --jazzy              # Start/enter container (Jazzy)
zobot_lab --prune --jazzy      # Remove container & image
zobot_lab --build --jazzy      # Rebuild image

# Build System
cr ba                          # Build all packages
cr b my_pkg                    # Build specific package
cr ca                          # Clean build artifacts
cr p                           # List packages

# VCS Management
ros-mgr import                 # Import repositories
ros-mgr build                  # Build repositories
ros-mgr install                # Install dependencies

# Requirements
req update                     # Update all dependencies

# Conda
activate_conda myenv           # Activate conda env
conda env list                 # List environments

# Navigation
ws                             # Go to workspace
rosws                          # Go to ros packages
```

### Important Paths

```
/home/zozo/zobot_ws            # Workspace (mounted from host)
/home/zozo/.ros_builds         # Build artifacts
/home/zozo/.anaconda           # Conda installation
/opt/ros/ros_ws                # VCS overlay workspace
/mnt/cave/.conda_docker        # Persistent conda backup
```

### Environment Variables

```bash
ROS_DISTRO=jazzy               # ROS distribution
ROS_DOMAIN_ID=0                # ROS domain
NVIDIA_VISIBLE_DEVICES=all     # GPU access
RCUTILS_COLORIZED_OUTPUT=1     # Colored logs
MAX_JOBS=8                     # Parallel jobs
```

---

**End of README**


