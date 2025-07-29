# Run Qwen3-Coder Locally

This repository contains scripts to automate the process of setting up and running the Qwen3-Coder-480B-A35B model locally using Unsloth Dynamic quants.

## Prerequisites

- A Linux system with an NVIDIA GPU.
- `sudo` privileges to install system dependencies.

## Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Qwen3-Coder-480B-A35B-local.git
    cd Qwen3-Coder-480B-A35B-local
    ```

2.  **Make the `run.sh` script executable:**
    ```bash
    chmod +x run.sh
    ```

3.  **Execute the script:**
    ```bash
    ./run.sh
    ```

The `run.sh` script will:
- Install necessary system dependencies (`build-essential`, `cmake`, `curl`, `libcurl4-openssl-dev`, `python3-pip`).
- Install required Python packages (`huggingface_hub`, `hf_transfer`).
- Clone the `llama.cpp` repository.
- Build `llama.cpp` with CUDA support.
- Download the Qwen3-Coder-480B-A35B-Instruct-GGUF model.
- Run the model with the recommended settings.

The script will prompt for your password to install system packages using `sudo`.

## Scripts

- `run.sh`: The main script to execute. It handles dependency installation and runs the Python script.
- `run_qwen.py`: A Python script that automates the process of cloning `llama.cpp`, building it, downloading the model, and running it.