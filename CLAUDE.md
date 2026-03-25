# Zobot Lab Workspace

This is a DevContainer-based robotics development workspace. The working directory on the host is `/lake/workspaces/env_ws`, which mounts to `/home/zozo/zobot_ws` inside the container.

## Architecture

Two Docker images:
- **Base** (`ghcr.io/snknitheesh/zobot-lab-jazzy-base:latest`) — CUDA 13.2, Python 3.12.12, PyTorch 2.11.0, TensorRT 10.16, TF, VPI4, ROS2 Jazzy, Gazebo Harmonic, Conda, Pixi, UV, PKL. Defined in `.devcontainer/Dockerfile.base`. Rarely rebuilt.
- **Overlay** (`zobot-lab-jazzy:latest`) — user setup, conda config, pip deps, VCS repos, workspace scripts. Defined in `.devcontainer/Dockerfile.ai`. Fast rebuilds.

## Container Management (`zobot_lab`)

Run from the **host**:
```
zobot_lab                  # Enter container
zobot_lab --pull           # Pull base from GHCR + build overlay
zobot_lab --build          # Build base + overlay locally
zobot_lab --rebuild        # Full rebuild, no cache
zobot_lab --prune          # Stop/remove containers (keeps images)
zobot_lab --conda <env>    # Enter with conda env activated
zobot_lab --vscode         # Open in VS Code DevContainer
```

## Inside the Container

### Building ROS2 Packages

Use `cr` (colcon runner) — **never use raw `colcon` commands**:
```
cr b                    # Build all packages in ros/
cr b my_package         # Build specific package
cr b pkg1 pkg2          # Build multiple packages
cr bf /opt/ros/ros_ws   # Build packages from a specific folder
cr t                    # Test all
cr t my_package         # Test specific package
cr c                    # Clean all build artifacts
cr p                    # List all available packages
cr h                    # Help
```

Shortcut aliases: `crb` (build all), `crt` (test all), `crc` (clean), `crp` (list), `crh` (help).

Build output goes to `~/.ros_builds/` (build, install, log dirs), NOT the workspace directory. Source code lives in `ros/`.

### Installing Dependencies

Use `req` to update all requirements from `.requirements/` files:
```
req update              # Install all deps (apt, pip, VCS repos)
req help                # Show help
```

Requirement files in `.requirements/`:
- `additional_pip_requirements.txt` — pip packages (baked into overlay at build time)
- `additional_ros_requirements.txt` — ROS apt packages
- `additional_deb_requirements.txt` — general apt packages
- `depend.repos` — VCS repos cloned to `/opt/ros/ros_ws` and built as overlay

### Managing VCS Repos

Use `ros-mgr` for external ROS dependency repos:
```
ros-mgr import          # Clone repos from depend.repos → /opt/ros/ros_ws
ros-mgr build           # Build imported repos
ros-mgr install         # rosdep install for all workspaces
ros-mgr pull            # Pull latest changes
ros-mgr status          # Show VCS status
ros-mgr clean           # Remove repos and build artifacts
```

### Navigation Aliases

```
ws                      # cd to workspace root
roscd                   # cd to ros/ packages directory
logs                    # cd to build logs
builds                  # cd to build output
cdmount                 # cd to mount directory
```

### ROS2 Aliases

```
rostopic                # ros2 topic
rosnode                 # ros2 node
rosparam                # ros2 param
rosservice              # ros2 service
rosrun                  # ros2 run
roslaunch               # ros2 launch
rosenv                  # show ROS environment variables
```

### System Aliases

```
gpu                     # nvidia-smi
meminfo                 # free -h
cpuinfo                 # lscpu
usage                   # disk usage summary
ports                   # show listening ports
```

## Key Directories

| Path (container) | Purpose |
|---|---|
| `/home/zozo/zobot_ws/` | Workspace root (mounted from host) |
| `/home/zozo/zobot_ws/ros/` | ROS2 source packages — put packages here |
| `/home/zozo/zobot_ws/debug/` | Debug/test ROS2 packages |
| `/home/zozo/zobot_ws/.bin/` | CLI tools (cr, req, ros-mgr, zobot_lab) |
| `/home/zozo/zobot_ws/.dev/` | Dev scripts (alias.sh, entrypoint) |
| `/home/zozo/zobot_ws/.requirements/` | Dependency files |
| `/home/zozo/.ros_builds/` | Colcon build/install/log output |
| `/opt/ros/jazzy/` | ROS2 Jazzy system install |
| `/opt/ros/ros_ws/` | VCS imported overlay repos |
| `/opt/conda/` | Conda installation |

## Key Paths (host)

| Path | Purpose |
|---|---|
| `/lake/workspaces/env_ws/` | This workspace root |
| `/lake/workspaces/env_ws/.devcontainer/Dockerfile.base` | Base image definition |
| `/lake/workspaces/env_ws/.devcontainer/Dockerfile.ai` | Overlay image definition |
| `/lake/workspaces/env_ws/docker-compose.yml` | Compose config |
| `/lake/.conda_docker/` | Persistent conda environments |

## ROS2 Environment

- **Distro:** Jazzy
- **RMW:** Zenoh (`rmw_zenoh_cpp`)
- **Domain ID:** 1
- **Python:** 3.12.12 (at `/usr/local/bin/python3`, system python at `/usr/bin/python3`)

## Git

- Do NOT add `Co-Authored-By` lines to commits
- Git user: `snknitheesh` / `snknitheesh@gmail.com`

## Conventions

- ROS2 packages go in `ros/` directory
- Use `cr b <pkg>` to build, never raw colcon
- Use `req update` to install deps, never manually pip/apt install for persistent deps
- Add pip deps to `.requirements/additional_pip_requirements.txt`
- Add ROS apt deps to `.requirements/additional_ros_requirements.txt`
- Add external ROS repos to `.requirements/depend.repos`
- Colcon defaults use `--symlink-install` and `Release` build type
- Container user is `zozo` (uid 1000)
- Container hostname is `ros2-jazzy`
- Network mode is `host` (no port mapping needed)
