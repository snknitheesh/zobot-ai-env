# Crypto Workspace Integration

**Last Updated:** December 25, 2025

## Overview

This document describes the integration of the `money_ws` (crypto prediction workspace) into the `env_ws` Docker container ecosystem. The crypto workspace is mounted as `/home/zozo/crypto_ws` inside the container and has its own isolated colcon build system to prevent interference with the main ROS workspace.

## Key Features

- **Isolated Build System**: Separate build, install, and log directories (`/home/zozo/.crypto_builds/`)
- **Dedicated Runner**: Custom colcon runner specifically for crypto workspace
- **Separate Configuration**: Independent colcon defaults to avoid conflicts
- **Volume Mount**: Persistent mount of host `money_ws` as container `crypto_ws`
- **Auto-Sourcing**: Automatic workspace sourcing after successful builds

## Directory Structure

```
Host System:
/home/zozo/workspaces/money_ws/              # Host directory
  └── crypto_prediction_ws/
      └── src/                                # ROS2 packages

Container System:
/home/zozo/crypto_ws/                        # Mounted workspace
  └── crypto_prediction_ws/
      └── src/                                # ROS2 packages

/home/zozo/.crypto_builds/                   # Build artifacts (separate from ROS)
  ├── build/                                 # Build files
  ├── install/                               # Installation files
  └── log/                                   # Build logs

/home/zozo/zobot_ws/.bin/                    # Management scripts
  ├── crypto_colcon_runner.py               # Crypto-specific colcon runner
  └── colcon_defaults_crypto.yaml           # Crypto workspace configuration
```

## Volume Mount Configuration

The `money_ws` directory is mounted in both startup scenarios:

### In `zobot_lab` Script (lines 177 & 216)

```bash
--volume "/home/zozo/workspaces/money_ws:/home/zozo/crypto_ws:rw"
```

This mount:
- Maps host `money_ws` to container `crypto_ws`
- Provides read-write access
- Persists across container restarts
- Is automatically included in all container starts

## Build System Configuration

### Colcon Defaults (`colcon_defaults_crypto.yaml`)

```yaml
build:
  symlink-install: true
  cmake-args:
    - "-DCMAKE_BUILD_TYPE=Release"
    - "-DBUILD_TESTING=OFF"
  base-paths:
    - "/home/zozo/crypto_ws/crypto_prediction_ws/src"
  build-base: "/home/zozo/.crypto_builds/build"
  install-base: "/home/zozo/.crypto_builds/install"
  log-base: "/home/zozo/.crypto_builds/log"
```

**Key Differences from Main Workspace:**
- Different `base-paths`: Points to `crypto_ws` instead of `zobot_ws/ros`
- Separate build directories: Uses `.crypto_builds` instead of `.ros_builds`
- Isolated configuration: No interference with main ROS workspace

## Usage

### Primary Command

```bash
cr crypto [COMMAND] [OPTIONS] [PACKAGES...]
```

### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `b`, `ba` | Build all packages | `cr crypto ba` |
| `b <pkg>...` | Build specific package(s) | `cr crypto b crypto_prediction` |
| `t`, `ta` | Test all packages | `cr crypto ta` |
| `t <pkg>...` | Test specific package(s) | `cr crypto t crypto_prediction` |
| `c`, `ca` | Clean all build artifacts | `cr crypto ca` |
| `p` | List all available packages | `cr crypto p` |
| `h` | Show help information | `cr crypto h` |

### Convenient Aliases

Defined in `alias.sh`:

```bash
crcrypto    # Direct access to crypto runner
crcb        # Build all crypto packages (crypto b)
crct        # Test all crypto packages (crypto t)
crcc        # Clean crypto builds (crypto c)
crcp        # List crypto packages (crypto p)
crch        # Show crypto help (crypto h)
cryptows    # Navigate to crypto workspace (cd /home/zozo/crypto_ws)
```

### Example Workflows

#### 1. Build All Crypto Packages

```bash
# Using full command
cr crypto ba

# Using alias
crcb

# Both methods:
# - Build all packages in /home/zozo/crypto_ws/crypto_prediction_ws/src
# - Output to /home/zozo/.crypto_builds/
# - Auto-source setup.bash on success
```

#### 2. Build Specific Package

```bash
cr crypto b crypto_prediction
```

#### 3. Clean and Rebuild

```bash
cr crypto ca  # Clean
cr crypto ba  # Build all
```

#### 4. List Available Packages

```bash
cr crypto p
# Output:
# Available crypto ROS packages:
#   1. crypto_prediction
```

#### 5. Navigate to Workspace

```bash
cryptows  # cd /home/zozo/crypto_ws
```

## Integration with Main Colcon Runner

The main colcon runner (`colcon_runner.py`) has been updated to recognize the `crypto` command and automatically delegate to `crypto_colcon_runner.py`:

```python
# In colcon_runner.py main()
if command == 'crypto':
    crypto_runner_path = Path(__file__).parent / "crypto_colcon_runner.py"
    if crypto_runner_path.exists():
        # Pass remaining args to crypto runner
        args = sys.argv[2:] if len(sys.argv) > 2 else []
        cmd = [str(crypto_runner_path)] + args
        result = subprocess.run(cmd)
        return result.returncode
```

This allows seamless switching between workspaces:

```bash
cr ba           # Build main ROS workspace
cr crypto ba    # Build crypto workspace
```

## Isolation Benefits

### 1. **No Build Interference**
- Crypto builds use `/home/zozo/.crypto_builds/`
- Main ROS builds use `/home/zozo/.ros_builds/`
- No risk of one workspace affecting the other

### 2. **Independent Configuration**
- Separate colcon defaults
- Different CMake arguments if needed
- Isolated testing configurations

### 3. **Clear Workspace Separation**
- Crypto packages in `crypto_ws/crypto_prediction_ws/src`
- ROS packages in `zobot_ws/ros`
- Easy to identify which workspace you're working with

### 4. **Parallel Development**
- Work on both workspaces simultaneously
- Independent build states
- No cleanup required when switching

## Setup in Dockerfile

The Dockerfile has been updated to include crypto-specific files:

```dockerfile
COPY .bin/colcon_defaults_crypto.yaml /home/zozo/zobot_ws/.bin/colcon_defaults_crypto.yaml
COPY .bin/crypto_colcon_runner.py /home/zozo/zobot_ws/.bin/crypto_colcon_runner.py
RUN chmod +x /home/zozo/zobot_ws/.bin/crypto_colcon_runner.py
```

**Note:** After modifying these files, you need to rebuild the container:

```bash
zobot_lab --build --jazzy  # or --humble
```

## Environment Sourcing

After a successful build, the crypto workspace needs to be sourced:

```bash
source /home/zozo/.crypto_builds/install/setup.bash
```

This is automatically suggested by the runner after each successful build.

To source both workspaces:

```bash
# Source ROS underlay
source /opt/ros/${ROS_DISTRO}/setup.bash

# Source main workspace
source /home/zozo/.ros_builds/install/setup.bash

# Source crypto workspace
source /home/zozo/.crypto_builds/install/setup.bash
```

## Troubleshooting

### Issue: "No ROS packages found in crypto_ws/src folder"

**Solution:**
```bash
# Verify mount exists
ls -la /home/zozo/crypto_ws/

# Check package structure
ls -la /home/zozo/crypto_ws/crypto_prediction_ws/src/
```

### Issue: "Crypto runner not found!"

**Solution:**
```bash
# Verify runner is installed
ls -la /home/zozo/zobot_ws/.bin/crypto_colcon_runner.py

# Check permissions
chmod +x /home/zozo/zobot_ws/.bin/crypto_colcon_runner.py
```

### Issue: Build artifacts filling up space

**Solution:**
```bash
# Clean crypto builds
cr crypto ca

# Clean both workspaces
cr ca          # Main workspace
cr crypto ca   # Crypto workspace
```

## Development Workflow

### Typical Session

```bash
# 1. Start container
zobot_lab --jazzy

# 2. Navigate to crypto workspace
cryptows

# 3. Make changes to packages
cd crypto_prediction_ws/src/crypto_prediction
# ... edit files ...

# 4. Build
crcb  # or: cr crypto ba

# 5. Source workspace
source /home/zozo/.crypto_builds/install/setup.bash

# 6. Test
cr crypto ta

# 7. Run nodes
ros2 run crypto_prediction <node_name>
```

### Working with Both Workspaces

```bash
# Terminal 1: Main ROS workspace
cd /home/zozo/zobot_ws
cr ba
source /home/zozo/.ros_builds/install/setup.bash

# Terminal 2: Crypto workspace
cryptows
crcb
source /home/zozo/.crypto_builds/install/setup.bash

# Both workspaces are now active in their respective terminals
```

## File Manifest

Files created/modified for crypto integration:

| File | Purpose | Location |
|------|---------|----------|
| `crypto_colcon_runner.py` | Main crypto build tool | `.bin/` |
| `colcon_defaults_crypto.yaml` | Crypto workspace config | `.bin/` |
| `alias.sh` | Added crypto aliases | `.bin/` (modified) |
| `colcon_runner.py` | Added crypto delegation | `.bin/` (modified) |
| `zobot_lab` | Added volume mount | `.bin/` (modified) |
| `Dockerfile.ai` | Copy crypto files | `.devcontainer/` (modified) |
| `CRYPTO_INTEGRATION.md` | This documentation | `.cursor/` |

## Best Practices

1. **Always use `cr crypto` commands** for crypto workspace operations
2. **Source the correct workspace** after building
3. **Use separate terminals** when working with both workspaces simultaneously
4. **Clean builds regularly** to save disk space
5. **Check package lists** with `cr crypto p` to verify setup

## Future Enhancements

Potential improvements for consideration:

- [ ] Auto-sourcing in `.bashrc` with workspace detection
- [ ] Build status indicators in prompt
- [ ] Integrated testing framework
- [ ] Workspace-specific environment variables
- [ ] Build time tracking and reporting
- [ ] Dependency management between workspaces

---

**Note:** This integration maintains the principle of separation of concerns, ensuring that the crypto prediction system remains isolated from the robotics development environment while still being easily accessible within the same container ecosystem.

