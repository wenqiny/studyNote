## Deepseek

This folder contains an `.py` file about some technologies which helps improve the performance for deepseek model.

The idea came from this [blog](https://medium.com/@atulit23/implementing-multi-head-latent-attention-from-scratch-in-python-1e14d03fbc91).

### MLA
MLA was used for lossy compress the $Q,\ K,\ V$ (they called **down-project**), then do decompress (they called **up-projection**) to finish the attention. It was inspired from **LoRA**, it shows there are much redundant dim in current AI model.
#### Benefits
1. KV cache (TODO for formula)
2. Matrix multiple in convert the hidden state $H$ into the $Q,\ K,\ V$. (TODO for formula)



Materials:
1. a [blog](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#mla-cache)
2. a [code case](https://medium.com/@atulit23/implementing-multi-head-latent-attention-from-scratch-in-python-1e14d03fbc91)