#!/usr/bin/env bash
# Install Python dependencies
pip install -r requirements.txt

# Compile the BuddyAllocation.c file
gcc BuddyAllocation.c -o BuddyAllocation -pthread

# Make the executable file executable
chmod +x BuddyAllocation

echo "Build completed successfully!"