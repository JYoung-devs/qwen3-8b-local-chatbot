from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "Qwen3-8B"
COMPUTE_DTYPE = torch.bfloat16


def load_chatbot():
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"Local model not found at {MODEL_PATH}. "
            "Place the Qwen3-8B files in models/Qwen3-8B."
        )

    # NF4 reduces VRAM use while BF16 keeps matrix operations GPU-friendly.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    print("Loading local Qwen model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=COMPUTE_DTYPE,
        local_files_only=True,
    )
    model.eval()

    return tokenizer, model


def print_runtime_info(model):
    architecture = (
        model.config.architectures[0]
        if model.config.architectures
        else model.__class__.__name__
    )
    parameter_count = model.num_parameters()
    input_device = model.get_input_embeddings().weight.device

    if input_device.type == "cuda":
        device_index = input_device.index or 0
        device_description = (
            f"{input_device} ({torch.cuda.get_device_name(device_index)})"
        )
    else:
        device_description = str(input_device)

    print("\nRuntime information")
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Architecture: {architecture} ({model.config.model_type})")
    print(f"  Parameters: {parameter_count:,} ({parameter_count / 1e9:.2f}B)")
    print("  Quantization: 4-bit NF4 (BitsAndBytes)")
    print(f"  Compute dtype: {str(COMPUTE_DTYPE).removeprefix('torch.')}")
    print(f"  Device: {device_description}")


def generate_response(tokenizer, model, messages):
    # Qwen's chat template disables generated reasoning while preserving chat roles.
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    input_device = model.get_input_embeddings().weight.device
    inputs = inputs.to(input_device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
        )

    # Decode only the new tokens, not the prompt or conversation history.
    prompt_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0, prompt_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def chat_loop(tokenizer, model):
    messages = []
    print("\nQwen ready. Type 'quit' or 'exit' to stop.\n")

    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in {"quit", "exit"}:
                break
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})
            response = generate_response(tokenizer, model, messages)
            print(f"\nQwen: {response}\n")

            # Retain only the visible answer so follow-up turns have clean context.
            messages.append({"role": "assistant", "content": response})
    except (EOFError, KeyboardInterrupt):
        print()

    print("Goodbye.")


def main():
    tokenizer, model = load_chatbot()
    print_runtime_info(model)
    chat_loop(tokenizer, model)


if __name__ == "__main__":
    main()
