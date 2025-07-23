#!/bin/bash

# Set target directory
TARGET_DIR="datasets"
ZIP_NAME="3DEditBench.zip"
URL="https://storage.googleapis.com/stanford_neuroai_models/SpelkeNet/3DEditBench/3DEditBench.zip"

# Create directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Download the zip file
echo "Downloading 3DEditBench.zip..."
curl -L "$URL" -o "$TARGET_DIR/$ZIP_NAME"

# Unzip it
echo "Unzipping into $TARGET_DIR..."
unzip -q "$TARGET_DIR/$ZIP_NAME" -d "$TARGET_DIR"

# Remove the zip file
rm "$TARGET_DIR/$ZIP_NAME"
echo "Done. Extracted to $TARGET_DIR/3DEditBench"
