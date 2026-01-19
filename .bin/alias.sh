#!/bin/bash

# Dynamic workspace detection for aliases
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

# Export for use in subshells
export WORKSPACE_ROOT
export WORKSPACE_BIN="/home/zozo/zobot_ws/.bin"

alias cr="python3 /home/zozo/zobot_ws/.bin/colcon_runner.py"
alias req="/home/zozo/zobot_ws/.bin/req"
alias ros-mgr="/home/zozo/zobot_ws/.bin/ros-mgr"


# Colcon commands
alias crb="cr b"     # Build all
alias crt="cr t"     # Test all  
alias crc="cr c"     # Clean all
alias crp="cr p"     # List packages
alias crh="cr h"     # Help

# Crypto workspace specific aliases
alias crcrypto="python3 /home/zozo/zobot_ws/.bin/crypto_colcon_runner.py"
alias crcb="crcrypto b"      # Build all crypto packages
alias crct="crcrypto t"      # Test all crypto packages  
alias crcc="crcrypto c"      # Clean crypto builds
alias crcp="crcrypto p"      # List crypto packages
alias crch="crcrypto h"      # Crypto help
alias cryptows="cd /home/zozo/crypto_ws"  # Navigate to crypto workspace


# Development aliases
alias pycheck="python -m py_compile"
alias pytest="python -m pytest"
alias mypy="python -m mypy"

# Docker aliases
alias dps="docker ps"
alias dpa="docker ps -a"
alias di="docker images"
alias dstop="docker stop"
alias drm="docker rm"
alias drmi="docker rmi"

# System aliases
alias ports="sudo netstat -tulpn | grep LISTEN"
alias usage="du -h --max-depth=1 | sort -hr"
alias meminfo="free -h"
alias cpuinfo="lscpu"
alias gpu="nvidia-smi"

# ROS aliases
alias rosenv="env | grep ROS"
alias rostopic="ros2 topic"
alias rosnode="ros2 node"
alias rosparam="ros2 param"
alias rosservice="ros2 service"
alias rosrun="ros2 run"
alias roslaunch="ros2 launch"

# Workspace navigation
alias ws="cd /home/zozo/zobot_ws"
alias rosws="cd /home/zozo/zobot_ws/ros"
alias logs="cd /home/zozo/.ros_builds/log"
alias builds="cd /home/zozo/.ros_builds"
alias cdmount="cd /home/zozo/zobot_ws/mount"
alias cave="cd /home/zozo/cave"

# Lerobot X Jazzy workspace 
alias lerobot_source="source /home/zozo/zobot_ws/ros/lerobot_x_jazzy/setup_workspace.sh"