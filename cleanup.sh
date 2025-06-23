#!/bin/bash

# Cleanup script to fix disk space issues
echo "Cleaning up build artifacts..."

# Remove egg-info directories
find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove build and dist directories
rm -rf build dist

# Remove __pycache__ directories
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove .pyc files
find . -name "*.pyc" -delete 2>/dev/null || true

# Check disk usage
echo "Current disk usage:"
df -h

echo "Cleanup completed."
