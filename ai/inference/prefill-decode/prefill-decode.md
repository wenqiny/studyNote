# Prefill and decode
There is two stages in inference for LLM, one is **prefill**, which process whole input at once and produce **first output token**, another one is **decode**, which generate **one output token** in each iteration according to **last token** the LLM generated.

## Prefill
The **prefill** stage takes a `(B, N, D)` tensor as its input, its workflow looks like (omit embedding and lm head layer and much details, just transformer layer):

$$
I_{N \times D} \xrightarrow{W_q, W_k, W_v} (Q_{N \times D} @ K_{N \times D}^{\mathsf{T}}) @ V_{N \times D} \xrightarrow{FFN} O_{N \times D}
$$

After that, we will pick the last row of $O_{N \times D}$ as the output to predicate **the first output token**.

## Decode
The **decode** stage takes a `(B, 1, D)` tensor and `(B, N, D)` kv cache as its input, its workflow looks like:

$$
I_{1 \times D} \xrightarrow{W_q} (Q_{1 \times D} @ K_{N \times D}^{\mathsf{T}}) @ V_{N \times D} \xrightarrow{FFN} O_{1 \times D}
$$

The we could use $O_{1 \times D}$ to predict **next token** (it has only 1 dim).

## Why we need two different stages?
A problem came to my mind, why we need to do **martxi mutiply** for whole $Q_{N \times D}@K_{N \times D}^{\mathsf{T}}$ in **prefill**, because I understand we just need the last dim to do predict **the first token**, we could do it just like **decode**: $Q_{1 \times D}@K_{N \times D}^{\mathsf{T}}$, right?

The answer is: No! There are two majority raeason:
1. We need to **fill all KV cache** in each transformer layer, so we need to calculated all K and V.
2. If we would like to calculate all K and V, we couldn't just do **matmul** for last token, let's image we do same thing as **decode**:

$$
I_{1 \times D} \xrightarrow{W_q} (Q_{1 \times D} @ K_{N \times D}^{\mathsf{T}}) @ V_{N \times D} \xrightarrow{FFN} O_{1 \times D}
$$

> we just get a $O_{1 \times D}$ finally, then we couldn't use it as **next layer's input** to get $Q_{N \times D},K_{N \times D},V_{N \times D}$ , because we just have **one dimesion**.


## Prefill output
Given we just use last output from the prefill, just for curousity, what are the tokens for the whole prefill out put?

Let's see an example for prompt with `You're Qwen, an AI assistant. Who are you?` for `Qwen3-4b` (source code at [here](./prefill-decode.py)):
```
prefill text is:
  show givenwen, a AI assistant. Please are you? What
final output:
 You're Qwen, an AI assistant. Who are you? What can you do? What are your capabilities? What are your limitations? Please answer in Chinese.
```

We could see at the end of the prefill output, these tokens are `are you? What`, they're same with source prompt, I guess maybe if we have a larger model we could get more same tokens as prompt.

Therefore there is another idea come! It's **speculative decoding**!