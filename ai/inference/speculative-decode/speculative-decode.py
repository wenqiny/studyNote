import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from typing import Tuple, Union
from transformers.cache_utils import DynamicCache

device = "cuda"

draft_model_name = "Qwen/Qwen3-0.6b"
traget_model_name = "Qwen/Qwen3-1.7b"
gamma = 5

def prune_tuple_cache(cache, num_tokens_to_discard):
    """
    Prune the cache by removing the specified number of tokens from the end. This pruning works for most models.
    It works for models having past_key_values such as Tuple of tuple(Tensor) of length n_layers, containing 2 or 4 tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head)

    Args:
        cache Tuple(Tuple[Tensor, Tensor]): The KV cache to be pruned.
        num_tokens_to_discard (int): The number of tokens to discard from the end of the cache.

    Returns:
        Tuple[Tensor, Tensor]: The pruned KV cache.
    """
    if cache is None:
        return None

    new_cache = []
    for layer_cache in cache:
        if layer_cache is None:
            new_cache.append(None)
            continue

        layer = []
        for i in range(len(layer_cache)):
            tensor = layer_cache[i]
            new_tensor = tensor[:, :, :-num_tokens_to_discard, :]
            layer.append(new_tensor)
        new_cache.append(tuple(layer))

    return tuple(new_cache)


def prune_dynamic_cache(cache, num_tokens_to_discard):
    """
    Prune the cache by removing the specified number of tokens from the end. This pruning works for models using DynamicCache.

    Args:
        cache (DynamicCache): The KV cache to be pruned.
        num_tokens_to_discard (int): The number of tokens to discard from the end of the cache.

    Returns:
        DynamicCache: The pruned KV cache. (same instance as the input cache, but modified in place)
    """
    if cache is None:
        return None

    for layer in range(len(cache)):
        cache.key_cache[layer] = cache.key_cache[layer][:, :, :-num_tokens_to_discard, :]
        cache.value_cache[layer] = cache.value_cache[layer][:, :, :-num_tokens_to_discard, :]
    cache._seen_tokens -= num_tokens_to_discard

    return cache

def prune_cache(cache, num_tokens_to_discard):
    """
    Prune the cache by removing the specified number of tokens from the end.

    Args:
        cache (Union[Tuple[Tuple[Tensor, Tensor]], DynamicCache]): The KV cache to be pruned.
        num_tokens_to_discard (int): The number of tokens to discard from the end of the cache.

    Returns:
        Union[Tuple[Tuple[Tensor, Tensor]], DynamicCache]: The pruned KV cache.
    """
    if cache is None:
        return None
    if isinstance(cache, tuple):
        return prune_tuple_cache(cache, num_tokens_to_discard)
    elif isinstance(cache, DynamicCache):
        return prune_dynamic_cache(cache, num_tokens_to_discard)
    else:
        raise ValueError("Unsupported cache type.")


tokenizer = AutoTokenizer.from_pretrained(traget_model_name) # both model have same vocabulary size, we could use same tokenizer
draft_model = AutoModelForCausalLM.from_pretrained(
    draft_model_name,
    device_map=device,
    torch_dtype=torch.float16
).eval()
target_model = AutoModelForCausalLM.from_pretrained(
    traget_model_name,
    device_map=device,
    torch_dtype=torch.float16
).eval()
vocab_size = target_model.get_input_embeddings().num_embeddings

prompt = "Once upon a time, in a sprawling, futuristic metropolis powered by geothermal energy, a lone AI detective named Unit 734 stumbled upon a cryptic message left behind by a vanished scientist. The message, displayed on a dusty holographic projector, read: 'The key to the city's future lies not in its towering spires, but deep beneath its shimmering surface, where ancient currents whisper secrets of forgotten power. Only those who dare to descend into the silence will find the truth.' Unit 734, despite its logical programming, felt an inexplicable pull towards the city's forgotten depths. Continue the story, describing Unit 734's initial descent and the strange environment it encounters, aiming for about 75-100 words."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device) # (1, seq_len)
max_decode_length = 50


def base_benchmark(input_ids):
    prompt_length = input_ids.size(1)
    max_length = prompt_length + max_decode_length
    generated_ids = torch.zeros((input_ids.shape[0], max_length), dtype=torch.long, device=device) # (1, max_length)
    generated_ids[:,:prompt_length] = input_ids
    kv_cache = None
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.perf_counter()
        for curr in range(prompt_length, max_length):
            output = target_model(
                input_ids=generated_ids[..., :curr],
                past_key_values=kv_cache,
                use_cache=True
            )
            kv_cache = output.past_key_values
            output_logits = output.logits[:, -1, :]
            generated_ids[0, curr] = torch.argmax(output_logits, dim=-1)
            # TODO(Wenqin): handling eos here.
        torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"Time cost for base: {end - start:.6f} seconds")
        print(f"base tps: {max_decode_length / (end - start):.6f}")

    return generated_ids[:,prompt_length:max_length]


def speculative_benchmark(input_ids):
    # prefill
    prompt_length = input_ids.size(1)
    max_length = prompt_length + max_decode_length
    generated_ids = torch.zeros((input_ids.shape[0], max_length), dtype=torch.long, device=device) # (1, max_length)
    generated_ids[:,:prompt_length] = input_ids
    draft_cache, target_cache = None, None
    with torch.no_grad():
        # prefill
        target_output = target_model(
            input_ids=generated_ids[..., :prompt_length],
            past_key_values=target_cache,
            use_cache=True
        )
        target_cache = target_output.past_key_values
        target_output_logits = target_output.logits[:, -1, :]
        generated_ids[0, prompt_length] = torch.argmax(target_output_logits, dim=-1)

        drafts_accepted, drafts_speculated = .0, .0

        # decode
        curr = prompt_length + 1
        torch.cuda.synchronize()
        start = time.perf_counter()
        while curr < max_length:
            print("curr: ", curr)
            corrected_gamma = min(gamma, max_length - curr)
            # draft_output=torch.zeros((1, corrected_gamma), dtype=torch.long, device=device)
            q = torch.zeros((1, corrected_gamma, vocab_size), device=device)
            for i in range(0, corrected_gamma):
                draft_output = target_model(
                    input_ids=generated_ids[..., :curr + i],
                    past_key_values=draft_cache,
                    use_cache=True
                )
                draft_cache = draft_output.past_key_values
                draft_output_logits = draft_output.logits[:, -1, :] # (1, vocabulary_size)
                generated_ids[0, curr + i] = torch.argmax(draft_output_logits, dim=-1)
                q[0, i]=draft_output_logits
            
            draft_cachegenerated_tokens = generated_ids[0, curr:curr+corrected_gamma]
            draft_cachegenerated_text = tokenizer.decode(draft_cachegenerated_tokens)
            print("draft: ", draft_cachegenerated_text)
            
            drafts_speculated += corrected_gamma

            # verify
            p = torch.zeros((1, corrected_gamma, vocab_size), device=device)
            target_verify_output = target_model(
                input_ids=generated_ids[..., :curr + corrected_gamma],
                past_key_values=target_cache,
                use_cache=True
            )
            target_verify_logits = target_verify_output.logits[:, -corrected_gamma:, :] # (1, corrected_gamma, vocabulary_size)
            target_cache=target_verify_output.past_key_values
            p[0] = target_verify_logits[0]

            target_generated_tokens = torch.argmax(target_verify_logits[0, :,:], dim=-1)
            target_generated_text = tokenizer.decode(target_generated_tokens)
            print("target: ", target_generated_text)

            r = torch.rand(corrected_gamma, device=device)
            p = torch.nn.functional.softmax(p, dim=-1) # p and q is logits, we want possibility
            q = torch.nn.functional.softmax(q, dim=-1)
            fraction = p / q
            n = corrected_gamma
            for i in range(corrected_gamma):
                if r[i] > fraction[0, i, generated_ids[0, curr+i]]:
                    # reject, because in such a case, the p is small than our expectation.
                    n = i
                    break

            drafts_accepted += n

            # TODO(Wenqin): handle eos token at here

            if n == corrected_gamma:
                # no tokens was rejected
                pass
            else:
                draft_cache = prune_cache(draft_cache, corrected_gamma - n)
                # why we + 1 here for prune target cache?
                # because we will accept the firt unmatch token for target model.
                target_cache = prune_cache(target_cache, corrected_gamma - n + 1)

                # We didn't need to modify any content in generated_ids, because
                # it use curr as mask in next iteration.

                # We just need to update the last token to take it as the output
                # of target model
                generated_ids[0, curr + n] = torch.argmax(target_verify_logits[:, -1, :], dim=-1)
            
            curr += n + 1

            # TODO(Wenqin): also handle eos token at here

        torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"Time cost for speculative: {end - start:.6f} seconds")
        print(f"speculative tps: {max_decode_length / (end - start)}")

    return generated_ids[:, prompt_length:max_length]


if __name__ == "__main__":
    # base_token = base_benchmark(input_ids)
    # base_text = tokenizer.decode(base_token[0], skip_special_tokens=True)
    # print("base text:\n", base_text)

    print("===================================")

    base_token = speculative_benchmark(input_ids)
    speculative_text = tokenizer.decode(base_token[0], skip_special_tokens=True)
    print("base text:\n", speculative_text)
