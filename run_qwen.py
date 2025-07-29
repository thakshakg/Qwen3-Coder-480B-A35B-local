import os
import subprocess

# --- Configuration ---
LLAMA_CPP_DIR = "llama.cpp"
MODEL_REPO_ID = "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF"
MODEL_FILE_PATTERN = "*UD-Q2_K_XL*"
MODEL_GGUF_FILE = "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF/UD-Q2_K_XL/Qwen3-Coder-480B-A35B-Instruct-UD-Q2_K_XL-00001-of-00004.gguf"

def run_command(command):
    """Executes a command and prints its output in real-time."""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc

def build_llama_cpp():
    """Clones and builds llama.cpp."""
    if not os.path.exists(LLAMA_CPP_DIR):
        print("--- Cloning llama.cpp ---")
        run_command(f"git clone https://github.com/ggml-org/llama.cpp {LLAMA_CPP_DIR}")

    print("--- Building llama.cpp ---")
    cmake_command = (
        f"cmake {LLAMA_CPP_DIR} -B {LLAMA_CPP_DIR}/build "
        "-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON"
    )
    build_command = f"cmake --build {LLAMA_CPP_DIR}/build --config Release -j --clean-first --target llama-cli llama-gguf-split"
    copy_command = f"cp {LLAMA_CPP_DIR}/build/bin/llama-* {LLAMA_CPP_DIR}"

    if run_command(cmake_command) != 0:
        print("!!! CMake configuration failed.")
        return
    if run_command(build_command) != 0:
        print("!!! Build failed.")
        return
    run_command(copy_command)
    print("--- llama.cpp built successfully ---")

def download_model():
    """Downloads the model from Hugging Face."""
    print("--- Downloading model ---")
    from huggingface_hub import snapshot_download
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    snapshot_download(
        repo_id=MODEL_REPO_ID,
        local_dir=os.path.dirname(MODEL_GGUF_FILE),
        allow_patterns=[MODEL_FILE_PATTERN],
    )
    print("--- Model downloaded successfully ---")

def run_model():
    """Runs the downloaded model."""
    print("--- Running model ---")
    run_command(
        f"./{LLAMA_CPP_DIR}/llama-cli "
        f"--model {MODEL_GGUF_FILE} "
        "--threads -1 "
        "--ctx-size 16384 "
        "--n-gpu-layers 99 "
        '-ot ".ffn_.*_exps.=CPU" '
        "--temp 0.7 "
        "--min-p 0.0 "
        "--top-p 0.8 "
        "--top-k 20 "
        "--repeat-penalty 1.05"
    )

if __name__ == "__main__":
    build_llama_cpp()
    download_model()
    run_model()
