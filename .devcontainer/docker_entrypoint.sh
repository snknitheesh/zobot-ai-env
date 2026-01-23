#!/bin/bash
# Custom entrypoint that shows packages instead of CUDA version

# Show installed packages (only if not shown before)
if [ ! -f /tmp/.packages_shown ]; then
    /usr/local/bin/show_packages.sh
    touch /tmp/.packages_shown
fi

# Execute the command passed to the container (default to bash if none provided)
exec "$@"
