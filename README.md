# Local Qwen3-8B Chatbot

A small multi-turn chatbot that runs Qwen3-8B locally with Hugging Face
Transformers and PyTorch. It loads the model with 4-bit NF4 quantization,
uses BF16 for computation, and keeps prior user and assistant messages as
conversation context.

## Architecture

```text
User text -> chat template/tokenizer -> Qwen3-8B -> PyTorch/CUDA -> NVIDIA GPU -> generated tokens
```

The application uses Qwen's chat-template switch to disable visible thinking
output during ordinary chat. Model and device details printed at startup come
from the loaded configuration and runtime rather than from generated text.

## Local model files

Model weights are **not included in this repository**. Place a complete local
Qwen3-8B model at:

```text
models/Qwen3-8B
```

The script sets `local_files_only=True`, so it does not download missing model
files. The `models/` directory and common weight formats are ignored by Git.
Do not force-add them or use Git LFS for this project.

## Windows setup

The existing environment can be activated and used directly:

```powershell
.\qwen-env\Scripts\Activate.ps1
python .\run_qwen.py
```

Alternatively, create a compatible environment:

```powershell
py -3.12 -m venv qwen-env
.\qwen-env\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

For NVIDIA acceleration, install the Windows CUDA build of PyTorch appropriate
for the installed driver using the official PyTorch installation selector.
CUDA-specific wheels may require a PyTorch package index, so the correct index
URL is intentionally not guessed here. The tested environment uses Python
3.12.10 and PyTorch 2.13.0+cu130.

After installing the appropriate PyTorch wheel, install the remaining pinned
dependencies:

```powershell
python -m pip install -r .\requirements.txt
```

The current runtime also requires an NVIDIA GPU supported by BitsAndBytes for
the configured 4-bit NF4 loading mode.

## Run

From the project directory:

```powershell
.\qwen-env\Scripts\python.exe .\run_qwen.py
```

Enter messages at the `You:` prompt. Conversation history is retained for
follow-up questions. Type `quit` or `exit` to stop.

The virtual environment, model files, caches, and `.env` files are ignored
because they are machine-specific, generated, very large, or potentially
sensitive. Only the source and reproducibility documentation belong in Git.

## Known warning

The tested Windows environment may print:

```text
triton not found; flop counting will not work for triton kernels
```

This warning concerns optional FLOP accounting. The BitsAndBytes CUDA backend,
model loading, and inference continue to work normally.
