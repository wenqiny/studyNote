import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.float16
).eval()

prompt = "You're Qwen, an AI assistant. Who are you?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda") # (1, seq_len)
past_kv = None

# Prefill
with torch.no_grad():
    prefill_output = model(input_ids=input_ids, use_cache=True)
    past_kv = prefill_output.past_key_values
    # print("prefill_output.logits.shape: ", prefill_output.logits.shape)
    next_token_logits = prefill_output.logits[:, -1, :] # logits shape [B, N+1, vocab_size]

prefill_output_tokens=torch.argmax(prefill_output.logits[:, 0:, :], dim=-1) # shape [batch, N+1, vocab_size] -> [batch, N+1]
print(prefill_output_tokens.shape)
prefill_text=tokenizer.decode(prefill_output_tokens[0]) # shape [N+1]
print("prefill text is:\n", prefill_text)

next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
generated_ids = torch.cat([input_ids, next_token], dim=-1)

max_new_tokens = 20

for _ in range(max_new_tokens):
    # TODO(Wenqin): try to handle EOS in token.
    last_token = generated_ids[:, -1:] # -1: keep the shape of last token as (1, 1) but not (1)
    with torch.no_grad():
        decode_output = model(input_ids=last_token, past_key_values=past_kv, use_cache=True)
        past_kv = decode_output.past_key_values
        # print("decode_output.logits.shape: ", decode_output.logits.shape)
        logits  = decode_output.logits[:, -1, :] # logits shape [B, 1, vocab_size]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    generated_ids = torch.cat([generated_ids, next_token], dim=-1)

output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("final output:\n", output_text)