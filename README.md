# 🤖 ZoBot AI - Robotics Development Environment

A robust, optimized Docker-based development environment for robotics development with ROS2 (Humble & Jazzy), CUDA support, and advanced tooling.

## ✨ Features

- **Optimized Multi-stage Docker builds** with perfect caching
- **Fixed user environment** (zozo) with full sudo privileges
- **Conda integration** with isolated environments
- **Advanced colcon runner** with hidden build directories
- **ROS2 Humble & Jazzy** support with easy switching
- **NVIDIA GPU support** with RTX 5090 optimized PyTorch
- **Dynamic requirements management** from within container
- **Volume mounting** capabilities for external data

## 🚀 Quick Start

### Prerequisites
- Docker with NVIDIA runtime
- NVIDIA drivers installed
- `/mnt/cave` directory (will be auto-mounted)

## 📋 Command Reference

### Container Management (zobot_ai)
| Command | Description |
|---------|-------------|
| `zobot_ai` | Start/enter Jazzy container |
| `zobot_ai --humble` | Start/enter Humble container |
| `zobot_ai --conda <env>` | Start with conda environment |
| `zobot_ai --volume <path>` | Mount external volume |
| `zobot_ai --build` | Build container only |
| `zobot_ai --prune` | Remove container and image |

### Colcon Runner (cr)
| Command | Description |
|---------|-------------|
| `cr ba` or `cr b` | Build all packages |
| `cr b <pkg1> <pkg2>` | Build specific packages |
| `cr ta` or `cr t` | Test all packages |
| `cr t <pkg1> <pkg2>` | Test specific packages |
| `cr ca` or `cr c` | Clean all build artifacts |
| `cr p` | List available packages |
| `cr h` | Show help |

### Requirements Management
| Command | Description |
|---------|-------------|
| `req update` | Update all requirements |
| `req help` | Show requirements help |

### Navigation & Utilities
| Alias | Command | Description |
|-------|---------|-------------|
| `rosws` | `cd /home/zozo/zobot_ws/ros` | Navigate to ROS workspace |
| `ws` | `cd /home/zozo/zobot_ws` | Navigate to main workspace |
| `cdmount` | `cd /home/zozo/mount` | Navigate to mounted volume |
| `cdcave` | `cd /home/zozo/cave` | Navigate to cave folder |
| `ca <env>` | `activate_conda <env>` | Activate conda environment |

## 📁 Directory Structure

```
/home/zozo/
├── zobot_ws/                 # Main workspace (mounted from host)
│   ├── ros/                  # ROS packages go here
│   ├── bin/                  # Scripts and tools
│   │   ├── zobot_ai          # Container management script
│   │   ├── colcon_runner.py  # Advanced colcon wrapper
│   │   ├── req               # Requirements manager
│   │   └── alias.sh          # Useful aliases
│   ├── requirements/         # Requirements files
│   │   ├── additional_pip_requirements.txt
│   │   ├── additional_deb_requirements.txt
│   │   └── additional_ros_requirements.txt
│   └── mount/               # External volume mount point
├── .ros_builds/             # Hidden build directory
│   ├── build/
│   ├── install/
│   └── log/
├── .anaconda/               # Conda installation
└── cave/                    # Cave folder mount from /mnt/cave
```

## 🔧 Development Workflow

### 1. Creating a ROS Package
```bash
# Enter container
./bin/zobot_ai

# Inside container: create package
cd /home/zozo/zobot_ws/ros
ros2 pkg create --build-type ament_cmake my_package

# Build the package
cr b my_package

```

### 2. Managing Dependencies
```bash
# Edit requirements files
nano /home/zozo/zobot_ws/requirements/additional_pip_requirements.txt

# Update requirements inside container
req update
```

### 3. Using Conda Environments
```bash
# List available environments
conda env list

# Create new environment
conda create -n robotics python=3.10

# Activate environment
ca robotics

# Or start container with environment
./bin/zobot_ai --conda robotics
```

### 4. Testing and Debugging
```bash
# Test specific package
cr t my_package

# Check build logs
cd /home/zozo/.ros_builds/log

# Clean and rebuild
cr c
cr b my_package
```

## 🐳 Container Features

### Multi-ROS Support
The environment supports both ROS2 Humble and Jazzy:
- Humble: `zobot_ai --humble` (default)
- Jazzy: `zobot_ai --jazzy`

### GPU Support
- Full NVIDIA GPU access
- PyTorch with CUDA 12.8 optimized for RTX 5090
- CUDA toolkit available

### Custom Volume Mounting
```bash
# Mount specific data directory
./bin/zobot_ai --volume /path/to/datasets

# Access mounted data inside container
cd /home/zozo/mount
```

**Happy Robot Building! 🤖✨**
