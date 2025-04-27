## KV cache
KV cache is a curcial feature in LLM inference.

### When KV cache was updated?
KV cache was updated before each layer of attention kernel was launched, at this moment, we could get an input tensor $X$, which was came from either previous layer's hidden state or the output for the embedding layer.

$X$ will be used for getting $Q$, $K$ and $V$, like:
$$
Q = W_qX\\
K = W_kX\\
V = W_vX
$$

After this computation, and before launch the attention kerne, like in `sglang` [code](https://github.com/sgl-project/sglang/blob/beb65c7433d6a5b8f72e5498200ee119d35476bf/python/sglang/srt/layers/attention/flashinfer_backend.py#L480-L485).

Then in the **launched kernel**, it will use the $K,\ V$ in the KV cache but not from the previous hidden states or embedding layer.