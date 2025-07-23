#!/bin/bash

# Set target directory
TARGET_DIR="datasets"
ZIP_NAME="precomputed_segments.zip"
URL="https://storage.googleapis.com/stanford_neuroai_models/SpelkeNet/3DEditBench/precomputed_segments.zip"

# Create directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Download the zip file
echo "Downloading precomputed_segments.zip..."
curl -L "$URL" -o "$TARGET_DIR/$ZIP_NAME"

# Unzip it
echo "Unzipping into $TARGET_DIR..."
unzip -q "$TARGET_DIR/$ZIP_NAME" -d "$TARGET_DIR"

# Remove the zip file
rm "$TARGET_DIR/$ZIP_NAME"
echo "Done. Extracted to $TARGET_DIR/precomputed_segments"
