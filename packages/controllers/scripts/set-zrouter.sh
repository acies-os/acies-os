#!/usr/bin/env sh

# Script to update ZROUTER value in .env file
# Usage: ./set-zrouter.sh <end-point>

# Check if end-point argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide an end-point value (format: <protocol>/<ip>:<port>)"
    echo "Usage: $0 <end-point>"
    exit 1
fi

END_POINT="$1"

# Check if .env file exists, create from template if not
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cat > .env << 'EOF'
ZROUTER=tcp/0.0.0.0:7447
EOF
    echo ".env file created successfully"
fi

# Use sed to replace the ZROUTER value
sed -i.bak "s|^ZROUTER=.*|ZROUTER=$END_POINT|" .env

# Check if the replacement was successful
if grep -q "ZROUTER=$END_POINT" .env; then
    echo "ZROUTER updated successfully to $END_POINT"
else
    echo "Error: Failed to update ZROUTER"
    exit 1
fi

# Remove the backup file created by sed
rm .env.bak
