import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _load_quantized_model(model_name: str):
    is_hip = getattr(torch.version, "hip", None) is not None
    bnb_available = False

    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes as bnb  # noqa: F401
        bnb_available = True
    except Exception:
        bnb_available = False

    if bnb_available:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        return model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=dtype)

    if device == "cuda":
        model = model.half()

    return model


def translate(prompt: str) -> str:
    model_name = "westenfelder/Llama-3.2-3B-Instruct-NL2SH"
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
    model = _load_quantized_model(model_name)

    messages = [
        {
            "role": "system",
            "content": "Your task is to translate a natural language instruction to a Bash command. You will receive an instruction in English and output a Bash command that can be run in a Linux terminal.",
        },
        {"role": "user", "content": f"{prompt}"},
    ]

    tokens = (
        tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
        .to(next(model.parameters()).device)
    )

    attention_mask = torch.ones_like(tokens)

    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    outputs = model.generate(
        tokens,
        attention_mask=attention_mask,
        max_new_tokens=100,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    response = outputs[0][tokens.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


if __name__ == "__main__":
    nl = "List files in the /workspace directory that were accessed over an hour ago."
    sh = translate(nl)
    print(sh)
