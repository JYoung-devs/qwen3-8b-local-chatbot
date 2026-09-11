# Local Qwen3-8B Chatbot

A small chatbot that runs Qwen3-8B locally with Hugging Face Transformers
and PyTorch. It loads the model with 4-bit NF4 quantization and uses BF16
for computation. The terminal chatbot retains conversation context; the
standalone browser prototype answers each message independently.

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

## Run the terminal chatbot

From the project directory:

```powershell
.\qwen-env\Scripts\python.exe .\run_qwen.py
```

Enter messages at the `You:` prompt. Conversation history is retained for
follow-up questions. Type `quit` or `exit` to stop.

The virtual environment, model files, caches, and `.env` files are ignored
because they are machine-specific, generated, very large, or potentially
sensitive. Only the source and reproducibility documentation belong in Git.

## Run the standalone web prototype

The browser prototype uses plain HTML, CSS, and JavaScript with no frontend
dependencies or build step. It stays separate from the Eiffel Tower school
website and uses no API keys or external AI service.

```text
run_qwen.py       Local model loading, generation, and terminal chat
server.py         FastAPI backend that reuses run_qwen.py
web/
  index.html     Chat page
  style.css      Page styling
  app.js         Browser requests and conversation display

Browser -> JavaScript fetch -> FastAPI -> local Qwen3-8B -> browser
```

Install the dependencies using the Windows setup instructions above. Then
open two PowerShell windows in the project directory.

In the first window, start the backend:

```powershell
.\qwen-env\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
```

Wait for `Application startup complete` and leave this window running.
The tokenizer and model load once at startup, and generation requests are
processed one at a time. Keep one worker and omit automatic reload to avoid
extra model loads. After changing `server.py`, stop and restart the backend.

In the second window, serve only the frontend directory:

```powershell
.\qwen-env\Scripts\python.exe -m http.server 5500 --bind 127.0.0.1 --directory web
```

Open [the local chat](http://127.0.0.1:5500/) in your browser. Use this HTTP
address so the page has an allowed origin. CORS allows only
`http://127.0.0.1:5500` and `http://localhost:5500`, with POST requests and
the Content-Type header. Both servers bind to this computer's loopback
address and should remain local during development.

Click **Send** or press **Enter** to submit; **Shift+Enter** adds a new line.
While Qwen generates, the controls are disabled and the page shows
`Qwen is thinking...`. Errors restore the message draft for retrying.
The visible conversation clears on refresh, and previous messages are not
sent as model context. Press **Ctrl+C** in each server window to stop it.

### Test the API directly

With the backend running, use another PowerShell window:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/chat" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"message":"Hello"}' |
  ConvertTo-Json
```

The request contains only a `message` string. The response has this shape;
Qwen's wording will vary:

```json
{"response": "Hello! How can I help?"}
```

## Known warning

The tested Windows environment may print:

```text
triton not found; flop counting will not work for triton kernels
```

This warning concerns optional FLOP accounting. The BitsAndBytes CUDA backend,
model loading, and inference continue to work normally.
