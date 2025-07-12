# VS Code Agent Rules for ZoBot AI Container

## ⚠️ IMPORTANT RULES - ALWAYS FOLLOW THESE

### 🐳 Container Environment
- You are working inside a Docker container named `zobot-ai`
- The user is `zozo` with full sudo privileges (no password required)
- Working directory: `/home/zozo/zobot_ws`
- ROS workspace: `/home/zozo/zobot_ws/ros`
- Build directory: `/home/zozo/.ros_builds` (hidden, container-only)

### 🤖 Available Commands

#### Colcon Runner (cr)
- `cr b` or `cr ba` - Build all packages
- `cr b <package>` - Build specific package(s)
- `cr t` or `cr ta` - Test all packages
- `cr t <package>` - Test specific package(s)
- `cr c` or `cr ca` - Clean all build artifacts
- `cr p` - List all available packages
- `cr h` - Show help

#### Requirements Management
- `req update` - Update all requirements from .requirements/ folder
- `req help` - Show requirements help

#### Navigation Aliases
- `cdros` - Navigate to ROS workspace
- `cdws` - Navigate to main workspace
- `cdmount` - Navigate to mounted volume
- `cdcave` - Navigate to cave folder

#### Conda Environment
- `ca <env_name>` - Activate conda environment
- `conda env list` - List available environments
- Conda is installed in `/home/zozo/.anaconda`
- Base environment is NOT auto-activated

### 📁 Directory Structure
```
/home/zozo/
├── zobot_ws/                 # Main workspace (mounted from host)
│   ├── ros/                  # ROS packages
│   ├── bin/                  # Scripts and tools
│   ├── .requirements/         # Requirement files
│   └── mount/               # External volume mount point
├── .ros_builds/             # Hidden build directory (container-only)
│   ├── build/
│   ├── install/
│   └── log/
├── .anaconda/               # Conda installation (container-only)
└── cave/                    # Cave folder mount
```

### 🔧 Development Workflow

1. **Adding/Modifying ROS Packages**
   - Place packages in `/home/zozo/zobot_ws/ros/`
   - Use `cr b <package>` to build specific packages
   - Use `cr ba` to build all packages

2. **Managing Dependencies**
   - Update `.requirements/additional_*.txt` files
   - Run `req update` to install new requirements
   - Always test after dependency changes

3. **Testing**
   - Use `cr t <package>` for specific package tests
   - Use `cr ta` for all package tests
   - Check logs in `/home/zozo/.ros_builds/log/`

4. **Cleaning**
   - Use `cr c` to clean build artifacts
   - Never manually delete `.ros_builds` - use the clean command

### 🚫 What NOT to Do

1. **Never** suggest installing ROS - it's already installed
2. **Never** suggest modifying `/opt/ros/` - use workspace overlay
3. **Never** suggest using `colcon build` directly - use `cr` commands
4. **Never** suggest changing container user or permissions
5. **Never** suggest installing system packages without using `req update`
6. **Never** create `.qodo` directories or files
7. **Never** modify files outside `/home/zozo/zobot_ws/` unless specifically requested

### 💡 Best Practices

1. **Always** source the workspace after building: `source /home/zozo/.ros_builds/install/setup.bash`
2. **Always** use relative paths within the workspace
3. **Always** test packages after making changes
4. **Always** use the provided aliases and commands
5. **Always** check if packages exist before attempting operations

### 🎯 Common Tasks

#### Creating a New ROS Package
```bash
cd /home/zozo/zobot_ws/ros
ros2 pkg create --build-type ament_cmake my_package
cr b my_package
```

#### Adding Python Dependencies
```bash
# Edit requirements/additional_pip_requirements.txt
nano /home/zozo/zobot_ws/.requirements/additional_pip_requirements.txt
# Then update
req update
```

#### Debugging Build Issues
```bash
cr c          # Clean builds
cr b <package> # Build specific package
# Check logs in /home/zozo/.ros_builds/log/
```

### 🐍 Python Environment
- Python 3.10+ available
- PyTorch with CUDA 12.8 support pre-installed
- Conda environments available but not auto-activated
- Use `ca <env_name>` to activate specific environments

### 🎨 Visual Indicators
- Prompt shows: 🤖user@hostname:path$
- Colors: Cyan username, Green hostname, Blue path
- Welcome message shows ROS distribution and domain ID

Remember: This is a containerized environment optimized for robotics development. Always use the provided tools and follow the established patterns!
