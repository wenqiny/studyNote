# Speculative decode
Speculative decode is a tech used in inference acceleration, its basic idea is use a **draft** (smaller) model to generate multiple tokens, then use a **traget** (larger) model to **verify** it.

The benefits came from that we could do **cheaper** inference from smaller model for **multiple tokens** and then verify them in larger model for **expensive** verification at **once**.

## Condition
Two models should meet below conditions:
1. Have same vocabulary size.

## pseudo code
1. Use **target** model to prefill kv cache and generate first token.
2. Feed kv cache and generated first token to **draft** model to do $\gamma$ iteration and get $\gamma$ tokens.
3. Use **target** with current kv cache and generated tokens to veryify, if verify passed, go to step 2, otherwise go to step 4.
4. Use the generated tokens from **target** model to update kv cache and generated tokens, then go to step 2.


## How to reject
We could get possibility from both draft and target model logits (after a softmax) as $p$ and $q$.

we could define $\text{accept rate} = \frac{p}{q}$, which means if $p$ is higher for the output token, and $q$ is lower for the output token, then we think the token is more likely to be accept!

It seems **a little unintuitive**, but we could though in two case:
1. If $p$ is very high and $q$ is very low, which means $\frac{p}{q}$ is very high, and in such a case, **draft with a lower possibility, it also get the correct result for target model, it's actually what we want**!!!
2. If $p$ is very low, and $q$ is very high, which means $\frac{p}{q}$ is very low, which means **draft model generate a token which the target model very unlikely to generate**,that's what we didn't want!!!

In another word, we could understand what we really want to see is $p$, because we already know draft model generate the output token, then the more $p$ is close to $1$ to more we will accept this token. 

Therefore we could introduce a `random` method to check whether we should reject or not:

$$
\text{rand(0, 1)} > \frac{p}{q}
$$

## My code

I tried to implement speculative decode [here](./speculative-decode.py), but it seems there is some wrong, the kv cache parts or something is wrong, which didn't generate correct result, try to **rewrite it later**.

## Reference
A good [code example](https://github.com/romsto/Speculative-Decoding)