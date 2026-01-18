# Container Execution Rules & Best Practices

**Last Updated:** December 25, 2025  
**Container:** zobot-jazzy (Docker)  
**Workspace:** `/home/zozo/zobot_ws`

---

## 🎯 Core Philosophy

**CRITICAL:** This container uses a **unified build management system** to maintain clean directory structure and prevent build/install/log artifacts from scattering across the workspace. **ALWAYS** use the provided tools and aliases.

---

## 📋 Mandatory Rules

### Rule 1: NEVER Use Raw `colcon` Commands

❌ **WRONG:**
```bash
# DO NOT DO THIS - Creates build/, install/, log/ in current directory
colcon build
colcon test
cd some_folder && colcon build
```

✅ **CORRECT:**
```bash
# ALWAYS use the colcon runner
cr ba              # Build all packages
cr b package_name  # Build specific package
cr crypto ba       # Build crypto workspace
```

**Why:** Raw `colcon` commands create `build/`, `install/`, and `log/` directories wherever you run them. The unified system ensures all artifacts go to dedicated locations:
- ROS packages → `/home/zozo/.ros_builds/`
- Crypto packages → `/home/zozo/.crypto_builds/`

---

### Rule 2: ALWAYS Use Aliases

The system provides optimized aliases for common operations. **Use them**.

#### Build & Test Aliases

**Main ROS Workspace (`zobot_ws`):**
```bash
cr ba    # Build all ROS packages
cr b     # Build specific package(s)
crb      # Shortcut for 'cr b'
cr ta    # Test all packages
cr t     # Test specific package(s)
crt      # Shortcut for 'cr t'
cr ca    # Clean all build artifacts
crc      # Shortcut for 'cr ca'
cr p     # List all packages
crp      # Shortcut for 'cr p'
cr h     # Show help
crh      # Shortcut for 'cr h'
```

**Crypto Workspace (`crypto_ws`):**
```bash
cr crypto ba    # Build all crypto packages
cr crypto b     # Build specific crypto package(s)
crcb            # Shortcut for crypto build all
cr crypto ta    # Test all crypto packages
cr crypto t     # Test specific crypto package(s)
crct            # Shortcut for crypto test all
cr crypto ca    # Clean crypto build artifacts
crcc            # Shortcut for crypto clean all
cr crypto p     # List all crypto packages
crcp            # Shortcut for crypto list packages
cr crypto h     # Show crypto help
crch            # Shortcut for crypto help
```

#### Workspace Navigation
```bash
ws          # → /home/zozo/zobot_ws
rosws       # → /home/zozo/zobot_ws/ros
cryptows    # → /home/zozo/crypto_ws
logs        # → /home/zozo/.ros_builds/log
builds      # → /home/zozo/.ros_builds
cave        # → /home/zozo/cave
cdmount     # → /home/zozo/zobot_ws/mount
```

#### Requirements Management
```bash
req         # Requirements manager (use 'req update')
ros-mgr     # ROS repository manager (VCS, rosdep)
```

#### System Monitoring
```bash
gpu         # nvidia-smi
ports       # Show listening ports
usage       # Disk usage
meminfo     # Memory info
```

---

### Rule 3: Understand the Build Directory Structure

```
/home/zozo/
├── zobot_ws/                    # Main workspace (volume-mounted)
│   ├── ros/                     # ROS packages source code
│   ├── .bin/                    # Build management scripts
│   ├── .requirements/           # Requirements files
│   └── mount/                   # Additional mounts
│
├── crypto_ws/                   # Crypto workspace (volume-mounted)
│   └── crypto_prediction_ws/    # Crypto packages
│       └── src/                 # Source code
│
├── .ros_builds/                 # ROS build artifacts (isolated)
│   ├── build/                   # Intermediate build files
│   ├── install/                 # Installed packages
│   └── log/                     # Build logs
│
└── .crypto_builds/              # Crypto build artifacts (isolated)
    ├── build/                   # Intermediate build files
    ├── install/                 # Installed packages
    └── log/                     # Build logs
```

**Why Separation?**
- **Clean workspace:** Source code directories stay clean
- **No conflicts:** ROS and crypto workspaces never interfere
- **Easy cleanup:** Delete build directories without affecting source
- **Persistent builds:** Builds survive workspace navigation

---

### Rule 4: Correct Build Workflow

#### For Main ROS Workspace

```bash
# 1. Navigate to workspace (optional, cr works from anywhere)
ws

# 2. Build all packages
cr ba

# 3. Build specific packages
cr b my_package another_package

# 4. Source the workspace
source ~/.ros_builds/install/setup.bash

# 5. Test packages
cr ta

# 6. Clean if needed
cr ca
```

#### For Crypto Workspace

```bash
# 1. Navigate to crypto workspace (optional)
cryptows

# 2. Build all crypto packages
cr crypto ba
# OR use alias
crcb

# 3. Build specific crypto packages
cr crypto b crypto_prediction

# 4. Source the crypto workspace
source ~/.crypto_builds/install/setup.bash

# 5. Test
cr crypto ta
# OR
crct

# 6. Clean if needed
cr crypto ca
# OR
crcc
```

#### For External/Overlay Packages

```bash
# Build packages from specific folder
cr bf /path/to/packages

# Example: Build VCS-imported packages
ros-mgr import   # Import from depend.repos
ros-mgr install  # Install dependencies
ros-mgr build    # Build imported packages (uses 'cr bf' internally)
```

---

### Rule 5: Package Development Workflow

#### Creating a New Package

```bash
# 1. Navigate to ROS source folder
rosws

# 2. Create package (Python or C++)
ros2 pkg create my_package --build-type ament_python
# OR
ros2 pkg create my_package --build-type ament_cmake

# 3. Develop your code...

# 4. Build the package
cr b my_package

# 5. Source workspace
source ~/.ros_builds/install/setup.bash

# 6. Test your package
ros2 run my_package my_node
```

#### Modifying Existing Package

```bash
# 1. Edit your code in ~/zobot_ws/ros/your_package/

# 2. Rebuild (only the modified package)
cr b your_package

# 3. Source workspace
source ~/.ros_builds/install/setup.bash

# 4. Test changes
ros2 run your_package your_node
```

---

### Rule 6: Requirements Management

#### Adding New Dependencies

**ROS Dependencies (apt packages):**
```bash
# 1. Edit requirements file
nano /home/zozo/zobot_ws/.requirements/additional_ros_requirements.txt

# 2. Add package name (one per line)
# Example: ros-jazzy-nav2-bringup

# 3. Update
req update
```

**Debian Packages:**
```bash
# 1. Edit requirements file
nano /home/zozo/zobot_ws/.requirements/additional_deb_requirements.txt

# 2. Add package name (one per line)
# Example: htop

# 3. Update
req update
```

**Python Packages:**
```bash
# 1. Edit requirements file
nano /home/zozo/zobot_ws/.requirements/additional_pip_requirements.txt

# 2. Add package name with version
# Example: numpy>=1.24.0

# 3. Update
req update
```

**VCS Repositories (Git repos):**
```bash
# 1. Edit depend.repos
nano /home/zozo/zobot_ws/.requirements/depend.repos

# 2. Add repository in YAML format
# Example:
# repositories:
#   some_repo:
#     type: git
#     url: https://github.com/user/repo.git
#     version: main

# 3. Import and build
ros-mgr import
ros-mgr install  # Install dependencies
ros-mgr build    # Build packages
```

---

### Rule 7: Sourcing Workspaces

**Critical:** After building, you must source the workspace for ROS to find your packages.

#### Manual Sourcing

```bash
# For main ROS workspace
source ~/.ros_builds/install/setup.bash

# For crypto workspace
source ~/.crypto_builds/install/setup.bash

# For both (order matters!)
source ~/.ros_builds/install/setup.bash
source ~/.crypto_builds/install/setup.bash
```

#### Auto-Sourcing in `.bashrc`

Add to your `.bashrc` inside container:
```bash
# Auto-source ROS workspace
if [ -f ~/.ros_builds/install/setup.bash ]; then
    source ~/.ros_builds/install/setup.bash
fi

# Auto-source crypto workspace (optional)
if [ -f ~/.crypto_builds/install/setup.bash ]; then
    source ~/.crypto_builds/install/setup.bash
fi
```

---

### Rule 8: Testing Best Practices

#### Run Tests

```bash
# Test all packages
cr ta

# Test specific package
cr t my_package

# View test results
cr ta  # Automatically shows results after test

# For crypto workspace
cr crypto ta
crct  # Alias
```

#### Test with Verbose Output

```bash
# Run colcon test with verbose flag
cd ~/.ros_builds
colcon test --packages-select my_package --event-handlers console_direct+
```

---

### Rule 9: Cleaning Build Artifacts

#### When to Clean

- After major dependency changes
- When experiencing strange build errors
- Before committing to save space
- When switching branches with different dependencies

#### How to Clean

```bash
# Clean ROS builds
cr ca
# OR
crc

# Clean crypto builds
cr crypto ca
# OR
crcc

# Clean both
cr ca && cr crypto ca
```

#### What Gets Cleaned

- `build/` - Intermediate build files
- `install/` - Installed packages
- `log/` - Build logs

**Note:** Cleaning does NOT affect source code in `~/zobot_ws/ros/` or `~/crypto_ws/`

---

### Rule 10: Working with Multiple Packages

#### Build Multiple Specific Packages

```bash
# Build several packages at once
cr b package1 package2 package3

# For crypto
cr crypto b pkg1 pkg2
```

#### Build Packages with Dependencies

```bash
# Build a package and all its dependencies
cr b my_package --packages-up-to my_package

# Note: Currently not supported by cr, use raw colcon in .ros_builds:
cd ~/.ros_builds
colcon build --packages-up-to my_package --symlink-install \
    --base-paths /home/zozo/zobot_ws/ros \
    --build-base /home/zozo/.ros_builds/build \
    --install-base /home/zozo/.ros_builds/install
```

---

## 🚀 Common Workflows

### Daily Development Workflow

```bash
# 1. Start container
zobot_lab --jazzy  # Run from host

# 2. Inside container - navigate to workspace
ws

# 3. Pull latest changes (if using git)
git pull

# 4. Update requirements if needed
req update

# 5. Build your packages
cr ba

# 6. Source workspace
source ~/.ros_builds/install/setup.bash

# 7. Develop and test iteratively
# - Edit code
# - cr b my_package
# - ros2 run my_package my_node
# - Test
# - Repeat
```

### Crypto Workspace Development

```bash
# 1. Navigate to crypto workspace
cryptows

# 2. Edit code in crypto_prediction_ws/src/

# 3. Build
crcb

# 4. Source
source ~/.crypto_builds/install/setup.bash

# 5. Run nodes
ros2 run crypto_prediction my_node

# 6. Test
crct
```

### Adding External Dependencies

```bash
# 1. Add to depend.repos
nano /home/zozo/zobot_ws/.requirements/depend.repos

# 2. Import repositories
ros-mgr import

# 3. Install dependencies
ros-mgr install

# 4. Build
ros-mgr build

# 5. Source
source ~/.ros_builds/install/setup.bash
```

### Fresh Start (Clean Rebuild)

```bash
# 1. Clean everything
cr ca
cr crypto ca

# 2. Rebuild everything
cr ba
cr crypto ba

# 3. Source workspaces
source ~/.ros_builds/install/setup.bash
source ~/.crypto_builds/install/setup.bash
```

---

## ⚠️ Common Mistakes to Avoid

### ❌ Mistake 1: Running `colcon build` in workspace root

```bash
cd /home/zozo/zobot_ws
colcon build  # WRONG! Creates build/, install/, log/ here
```

**Fix:** Use `cr ba` instead.

---

### ❌ Mistake 2: Not sourcing after build

```bash
cr ba
ros2 run my_package my_node  # ERROR: Package not found
```

**Fix:** Source the workspace first:
```bash
source ~/.ros_builds/install/setup.bash
ros2 run my_package my_node  # Now works
```

---

### ❌ Mistake 3: Building in wrong directory

```bash
cd /some/random/directory
colcon build --symlink-install  # WRONG! Creates artifacts here
```

**Fix:** Use `cr` which handles directories automatically:
```bash
cr ba  # Works from anywhere
```

---

### ❌ Mistake 4: Mixing build systems

```bash
# Building with raw colcon in workspace
cd ~/zobot_ws/ros
colcon build

# Then using cr
cr ba
# Now you have builds in TWO places! Conflicts!
```

**Fix:** Pick ONE system (use `cr`) and stick with it.

---

### ❌ Mistake 5: Forgetting workspace isolation

```bash
cr ba  # Builds ROS packages
cr crypto ba  # Builds crypto packages
source ~/.ros_builds/install/setup.bash  # Only ROS sourced
ros2 run crypto_prediction my_node  # ERROR: Package not found
```

**Fix:** Source both workspaces if you need both:
```bash
source ~/.ros_builds/install/setup.bash
source ~/.crypto_builds/install/setup.bash
```

---

## 🔍 Debugging Build Issues

### Check What's Being Built

```bash
# List available packages
cr p      # ROS packages
cr crypto p  # Crypto packages
```

### View Build Logs

```bash
# Navigate to log directory
logs  # Alias for cd ~/.ros_builds/log

# Find latest build
ls -lt | head

# View log
less build_YYYY-MM-DD_HH-MM-SS/events.log
```

### Verbose Build

```bash
# Build with more output
cd ~/.ros_builds
colcon build --symlink-install \
    --base-paths /home/zozo/zobot_ws/ros \
    --build-base /home/zozo/.ros_builds/build \
    --install-base /home/zozo/.ros_builds/install \
    --event-handlers console_direct+
```

### Check Package Dependencies

```bash
# Inside package directory
rosws
cd my_package
cat package.xml  # Check <depend> tags
```

### Verify Sourcing

```bash
# Check if workspace is sourced
echo $AMENT_PREFIX_PATH
# Should include ~/.ros_builds/install

# Check available packages
ros2 pkg list | grep my_package
```

---

## 📊 Build System Architecture

### Colcon Runner (`cr`)

**Location:** `/home/zozo/zobot_ws/.bin/colcon_runner.py`

**Features:**
- Automatic workspace detection
- Isolated build directories (`~/.ros_builds/`)
- Color-coded output
- Package validation
- Symlink install by default
- Release build by default

**Commands:**
- `ba` - Build all
- `b <pkg>...` - Build specific packages
- `bf <path>` - Build from folder
- `ta` - Test all
- `t <pkg>...` - Test specific
- `ca` - Clean all
- `p` - List packages
- `h` - Help

### Crypto Colcon Runner (`cr crypto`)

**Location:** `/home/zozo/zobot_ws/.bin/crypto_colcon_runner.py`

**Features:**
- Separate build directory (`~/.crypto_builds/`)
- Points to crypto workspace
- Same command interface as main runner
- Complete isolation from ROS builds

**Integration:**
- Called via `cr crypto <command>`
- Automatic delegation from main runner
- Separate aliases for convenience

### ROS Manager (`ros-mgr`)

**Location:** `/home/zozo/zobot_ws/.bin/ros-mgr`

**Features:**
- VCS repository management
- Rosdep dependency installation
- Uses `cr bf` for building external packages

**Commands:**
- `import` - Import from depend.repos
- `status` - Check repository status
- `pull` - Pull latest changes
- `build` - Build imported packages
- `install` - Install dependencies
- `clean` - Clean repositories

### Requirements Manager (`req`)

**Location:** `/home/zozo/zobot_ws/.bin/req`

**Features:**
- Unified requirements management
- Handles apt, pip, and VCS
- Integrates with ros-mgr

**Commands:**
- `update` - Update all requirements

---

## 🎓 Advanced Topics

### Custom Build Flags

If you need custom build flags, use `cr bf` with manual colcon:

```bash
cd ~/.ros_builds
colcon build --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Debug \
    --base-paths /home/zozo/zobot_ws/ros/my_package \
    --build-base /home/zozo/.ros_builds/build \
    --install-base /home/zozo/.ros_builds/install
```

### Parallel Builds

The runners use default parallelism. To control:

```bash
cd ~/.ros_builds
colcon build --symlink-install \
    --parallel-workers 4 \
    --base-paths /home/zozo/zobot_ws/ros \
    --build-base /home/zozo/.ros_builds/build \
    --install-base /home/zozo/.ros_builds/install
```

### Build Testing Enabled

```bash
cd ~/.ros_builds
colcon build --symlink-install \
    --cmake-args -DBUILD_TESTING=ON \
    --base-paths /home/zozo/zobot_ws/ros \
    --build-base /home/zozo/.ros_builds/build \
    --install-base /home/zozo/.ros_builds/install
```

---

## 📝 Quick Reference Card

```bash
# BUILD COMMANDS
cr ba              # Build all ROS packages
cr b <pkg>         # Build specific package
cr crypto ba       # Build all crypto packages
crcb               # Alias: Build all crypto packages

# TEST COMMANDS
cr ta              # Test all ROS packages
cr t <pkg>         # Test specific package
cr crypto ta       # Test all crypto packages
crct               # Alias: Test all crypto packages

# CLEAN COMMANDS
cr ca              # Clean ROS builds
cr crypto ca       # Clean crypto builds
crc                # Alias: Clean ROS builds
crcc               # Alias: Clean crypto builds

# LIST COMMANDS
cr p               # List ROS packages
cr crypto p        # List crypto packages

# REQUIREMENTS
req update         # Update all requirements

# ROS REPOS
ros-mgr import     # Import VCS repositories
ros-mgr install    # Install rosdep dependencies
ros-mgr build      # Build imported packages

# NAVIGATION
ws                 # → /home/zozo/zobot_ws
rosws              # → /home/zozo/zobot_ws/ros
cryptows           # → /home/zozo/crypto_ws
logs               # → ~/.ros_builds/log
builds             # → ~/.ros_builds

# SOURCING
source ~/.ros_builds/install/setup.bash      # Source ROS workspace
source ~/.crypto_builds/install/setup.bash   # Source crypto workspace

# SYSTEM
gpu                # nvidia-smi
ports              # Show listening ports
usage              # Disk usage
```

---

## 🔒 Critical Reminders

1. ✅ **ALWAYS use `cr` or aliases** - Never use raw `colcon` in workspace
2. ✅ **Source after building** - Workspace must be sourced to find packages
3. ✅ **Separate workspaces** - ROS and crypto are isolated, don't mix
4. ✅ **Use `req update`** - For adding dependencies, don't install manually
5. ✅ **Check `cr p`** - List packages to verify they're found
6. ✅ **Clean when stuck** - `cr ca` fixes most strange build issues
7. ✅ **Logs in dedicated directory** - Check `~/.ros_builds/log/`
8. ✅ **VCS repos via `ros-mgr`** - For external packages

---

## 🆘 Help & Support

### Get Help

```bash
cr h               # Colcon runner help
cr crypto h        # Crypto runner help
ros-mgr help       # ROS manager help
req help           # Requirements manager help
```

### Check System State

```bash
cr p               # List ROS packages
cr crypto p        # List crypto packages
builds             # Navigate to builds directory
logs               # Navigate to logs directory
gpu                # Check GPU status
```

---

**Remember:** The unified build system exists to keep your workspace clean and organized. Follow these rules and your development experience will be smooth and predictable!


