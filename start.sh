#!/bin/bash

# Save current directory
CUR_DIR="$(pwd)"

# Media Campaign Cost Prediction Project Setup
echo "Setting up Media Campaign Cost Prediction environment..."

# Create virtual environment
python -m venv venv

# Activate virtual environment
if [ -d "venv/bin" ]; then
  source venv/bin/activate
else
  source venv/Scripts/activate
fi

# Install dependencies
echo "Installing required packages..."
pip install -r requirements.txt

# Install Jupyter notebook
echo "Installing Jupyter notebook..."
pip install jupyter

# Create figures directory
echo "Creating figures directory..."
mkdir -p src/figures

# Check if datasets exist, otherwise provide instructions
echo "Checking for datasets..."
missing=0

if [ ! -f "data/train.csv" ]; then
  missing=1
fi
if [ ! -f "data/test.csv" ]; then
  missing=1
fi

if [ $missing -eq 1 ]; then
  echo "WARNING: One or more datasets are missing. Please ensure you have the following files in the project directory:"
  echo "- data/train.csv (360,336 samples)"
  echo "- data/test.csv (240,224 samples)"
fi

echo "Setup complete! To start working with the project:"
echo "1. The environment is now activated"
echo "2. To run the Python script: python src/assignment.py"
echo "3. To run the Jupyter notebook: jupyter notebook"
echo "4. To deactivate the virtual environment: deactivate"
