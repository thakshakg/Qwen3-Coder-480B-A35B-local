#!/bin/bash

# --- Install Dependencies ---
sudo apt-get update
sudo apt-get install -y build-essential cmake curl libcurl4-openssl-dev python3-pip

# --- Install Python Packages ---
pip3 install huggingface_hub hf_transfer

# --- Run the Python Script ---
python3 run_qwen.py
