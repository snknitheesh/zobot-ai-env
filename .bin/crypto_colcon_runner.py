#!/usr/bin/env python3
"""
Crypto Colcon Runner - ROS2 package build and test management for crypto_ws
Usage: cr crypto [COMMAND] [OPTIONS] [PACKAGES...]
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# ANSI color codes
class Colors:
    GREEN = '\033[0;32m'
    CYAN = '\033[0;36m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BOLD = '\033[1m'
    NC = '\033[0m'  

class CryptoColconRunner:
    def __init__(self):
        # Crypto workspace-specific paths
        self.workspace_dir = Path("/home/zozo/crypto_ws")
        self.home_dir = Path.home()
        self.src_folder = self.workspace_dir / "crypto_prediction" / "src"
        self.build_dir = self.home_dir / ".crypto_builds"
        self.colcon_defaults = Path("/home/zozo/zobot_ws/.bin/colcon_defaults_crypto.yaml")
        
        # Create necessary directories
        self.build_dir.mkdir(exist_ok=True, parents=True)
        
        self.ros_distro = os.environ.get('ROS_DISTRO', 'humble')
        
    def print_colored(self, message: str, color: str = Colors.NC):
        """Print colored message"""
        print(f"{color}{message}{Colors.NC}")
        
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> int:
        """Run shell command and return exit code"""
        if cwd is None:
            cwd = self.build_dir
            
        self.print_colored(f"🔧 Running: {' '.join(cmd)}", Colors.CYAN)
        
        try:
            result = subprocess.run(cmd, cwd=cwd, check=False)
            return result.returncode
        except Exception as e:
            self.print_colored(f"Error running command: {e}", Colors.RED)
            return 1
    
    def source_ros_setup(self):
        """Source ROS setup files"""
        setup_files = [
            f"/opt/ros/{self.ros_distro}/setup.bash",
            f"{self.build_dir}/install/setup.bash"
        ]
        
        source_cmd = []
        for setup_file in setup_files:
            if Path(setup_file).exists():
                source_cmd.append(f"source {setup_file}")
        
        if source_cmd:
            return " && ".join(source_cmd) + " && "
        return ""
    
    def get_packages(self) -> List[str]:
        """Get list of available ROS packages"""
        packages = []
        if self.src_folder.exists():
            for item in self.src_folder.iterdir():
                if item.is_dir() and (item / "package.xml").exists():
                    packages.append(item.name)
        return sorted(packages)
    
    def build_packages(self, packages: List[str] = None, test_mode: bool = False):
        """Build specified packages or all packages"""
        if not self.src_folder.exists():
            self.print_colored(f"Source folder does not exist: {self.src_folder}", Colors.RED)
            return 1
            
        if packages:
            available_packages = self.get_packages()
            invalid_packages = [pkg for pkg in packages if pkg not in available_packages]
            if invalid_packages:
                self.print_colored(f"Invalid packages: {', '.join(invalid_packages)}", Colors.RED)
                return 1
                
            package_args = ["--packages-select"] + packages
            self.print_colored(f"🔐 Building crypto packages: {', '.join(packages)}", Colors.GREEN)
        else:
            package_args = []
            self.print_colored("🔐 Building all crypto packages", Colors.GREEN)
        
        build_cmd = [
            "colcon", "build",
            "--symlink-install",
            "--cmake-args", "-DCMAKE_BUILD_TYPE=Release",
            f"-DBUILD_TESTING={'ON' if test_mode else 'OFF'}",
            "--base-paths", str(self.src_folder),
            "--build-base", str(self.build_dir / "build"),
            "--install-base", str(self.build_dir / "install"),
        ] + package_args
        
        result = self.run_command(build_cmd)
        
        if result == 0:
            self.print_colored("✅ Build successful!", Colors.GREEN)
            self.print_colored("Sourcing crypto workspace...", Colors.YELLOW)
            
            setup_path = self.build_dir / "install" / "setup.bash"
            if setup_path.exists():
                self.print_colored(f"Run: source {setup_path}", Colors.CYAN)
                # Auto-source in current shell environment
                os.system(f"bash -c 'source {setup_path}'")
        else:
            self.print_colored("❌ Build failed!", Colors.RED)
            
        return result
    
    def test_packages(self, packages: List[str] = None):
        """Test specified packages or all packages"""
        if packages:
            package_args = ["--packages-select"] + packages
            self.print_colored(f"Testing crypto packages: {', '.join(packages)}", Colors.GREEN)
        else:
            package_args = []
            self.print_colored("Testing all crypto packages", Colors.GREEN)
        
        test_cmd = [
            "colcon", "test",
            "--base-paths", str(self.src_folder),
            "--build-base", str(self.build_dir / "build"),
            "--install-base", str(self.build_dir / "install"),
        ] + package_args
        
        result = self.run_command(test_cmd)
        
        if result == 0:
            result_cmd = [
                "colcon", "test-result", "--verbose",
                "--test-result-base", str(self.build_dir / "log")
            ]
            self.run_command(result_cmd)
            
        return result
    
    def clean_build(self):
        """Clean all build artifacts"""
        self.print_colored("Cleaning crypto build artifacts...", Colors.YELLOW)
        
        for item in self.build_dir.iterdir():
            if item.is_dir():
                subprocess.run(["rm", "-rf", str(item)], check=False)
            else:
                item.unlink(missing_ok=True)
        
        self.print_colored("Crypto build artifacts cleaned!", Colors.GREEN)
        return 0
    
    def list_packages(self):
        """List all available packages"""
        packages = self.get_packages()
        
        if packages:
            self.print_colored("Available crypto ROS packages:", Colors.GREEN)
            for i, pkg in enumerate(packages, 1):
                print(f"  {i:2d}. {pkg}")
        else:
            self.print_colored("No ROS packages found in crypto_ws/src folder", Colors.YELLOW)
            
        return 0
    
    def show_help(self):
        """Show help information"""
        help_text = f"""
{Colors.CYAN}Crypto Colcon Runner - Crypto Prediction Workspace Management{Colors.NC}

{Colors.GREEN}Usage:{Colors.NC}
  cr crypto [COMMAND] [PACKAGES...]

{Colors.GREEN}Commands:{Colors.NC}
  b, ba          Build all packages
  b <pkg>...     Build specific package(s)
  t, ta          Test all packages  
  t <pkg>...     Test specific package(s)
  c, ca          Clean all build artifacts
  p              List all available packages
  h              Show this help

{Colors.GREEN}Examples:{Colors.NC}
  cr crypto ba                 # Build all crypto packages
  cr crypto b crypto_predict   # Build specific package
  cr crypto b pkg1 pkg2        # Build multiple packages
  cr crypto ta                 # Test all packages
  cr crypto t my_package       # Test specific package
  cr crypto ca                 # Clean all builds
  cr crypto p                  # List packages

{Colors.GREEN}Build Directory:{Colors.NC}
  {self.build_dir}

{Colors.GREEN}Source Directory:{Colors.NC}
  {self.src_folder}

{Colors.YELLOW}Note:{Colors.NC} This runner uses separate build directories to avoid
interference with the main ROS workspace (zobot_ws).
"""
        print(help_text)
        return 0

def main():
    runner = CryptoColconRunner()
    
    if len(sys.argv) < 2:
        runner.show_help()
        return 0
    
    command = sys.argv[1].lower()
    packages = sys.argv[2:] if len(sys.argv) > 2 else []
    
    try:
        if command in ['b', 'ba']:
            if command == 'ba' or not packages:
                return runner.build_packages()
            else:
                return runner.build_packages(packages)
                
        elif command in ['t', 'ta']:
            if command == 'ta' or not packages:
                return runner.test_packages()
            else:
                return runner.test_packages(packages)
                
        elif command in ['c', 'ca']:
            return runner.clean_build()
            
        elif command == 'p':
            return runner.list_packages()
            
        elif command == 'h':
            return runner.show_help()
            
        else:
            runner.print_colored(f"Unknown command: {command}", Colors.RED)
            runner.show_help()
            return 1
            
    except KeyboardInterrupt:
        runner.print_colored("\nOperation cancelled by user", Colors.YELLOW)
        return 1
    except Exception as e:
        runner.print_colored(f"Unexpected error: {e}", Colors.RED)
        return 1

if __name__ == "__main__":
    sys.exit(main())

