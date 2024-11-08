#!/bin/bash

# Set the URL and output directory
URL="http://www.openslr.org/resources/12/train-clean-100.tar.gz"
OUTPUT_DIR="../data/LibriSpeech/train-clean-100"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Download the dataset
echo "Downloading LibriSpeech train-clean-100 dataset..."
wget -c $URL -P $OUTPUT_DIR

# Extract the dataset
echo "Extracting the dataset..."
tar -xzf $OUTPUT_DIR/train-clean-100.tar.gz -C $OUTPUT_DIR

# Remove the tar.gz file after extraction
echo "Cleaning up..."
rm $OUTPUT_DIR/train-clean-100.tar.gz

echo "Download and extraction complete. Dataset is available in $OUTPUT_DIR"
