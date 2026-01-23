#!/bin/bash
# Custom entrypoint that shows packages on container start

# Show installed packages and mark as shown
/usr/local/bin/show_packages.sh
touch /tmp/.packages_shown

# Execute the command passed to the container (default to bash if none provided)
exec "$@"
