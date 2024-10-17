cache tag and index
==============
## cache kind
1. Directed map [[1](https://www.cs.cornell.edu/~tomf/notes/cps104/cache.html#direct)]
2. N-way associated [[2](https://www.cs.cornell.edu/~tomf/notes/cps104/cache.html#assoc)]
3. Full associated [[3](https://www.cs.cornell.edu/~tomf/notes/cps104/cache.html#full)]
It's expensive in cicurtal, hence just use for very small cache, like TLB

We use index and tag to identify a cache line:
There is a 8 Way cache with 16 set, each CL(cache line) is 64 bytes.
`Total size = 8 * 16 * 64 = 8192 bytes`
It looks like
```
|-------| Way 0 | Way 1 | ..... | Way 8 |
| Set 0 |   CL  |   CL  | ..... |   CL  |
| Set 1 |   CL  |   CL  | ..... |   CL  |
| ..... | ..... | ..... | ..... | ..... |
| Set16 |   CL  |   CL  | ..... |   CL  |
```

A CL looks like
```
| V | tag | data |
```
In side it, V means does this cache line valid, tag is the tag to check whether cache hit(it's related to the address), and data is the really data(64 bytes) for the CL.

In that case we needs 8 bits for `offset()64 bytes`, 4 bits for `set(16)`.

If we take 32 bits address, it looks like:
```
|---------|--------|--------|
|   tag   | index  | offset |
| 32 - 12 | 11 - 8 | 7 - 0  |
```

But what the address is? physical or logical?There is some ways to map memory address in to cache.

Bellow could refer to [wiki](https://en.wikipedia.org/wiki/CPU_cache#Address_translation)
1. PIPT [[4](https://en.wikipedia.org/wiki/CPU_cache#Cache_entry_structure:~:text=Physically%20indexed%2C%20physically,in%20the%20cache.)]
2. VIVT [[5](https://en.wikipedia.org/wiki/CPU_cache#Cache_entry_structure:~:text=Virtually%20indexed%2C%20virtually,physical%20addresses%20(VIPT).)]
3. VIPT [[6](https://en.wikipedia.org/wiki/CPU_cache#Address_translation:~:text=Virtually%20indexed%2C%20physically,of%20the%20cache.)]
3. PIVT [[7](https://en.wikipedia.org/wiki/CPU_cache#Address_translation:~:text=Physically%20indexed%2C%20virtually,the%20virtual%20address.)]


Except PIPT, other way may introduce [homonmy and sononym](https://en.wikipedia.org/wiki/CPU_cache#Homonym_and_synonym_problems). But if we keep `index bits + offset bits < page size bits` we could take VIPT as same as PIPT, because the VI is same as PI, but it may not suitable for big cache(just for L1 cache normally).

It looks [page coloring](https://en.wikipedia.org/wiki/CPU_cache#Page_coloring) could address this proble form end of this [blog](https://zhuanlan.zhihu.com/p/577138649).

It said
```
如果你留心市面上最新的主流处理器的话，你会发现他们的L1D都遵循了这个规律。比如：

intel 的sunny cove L1D： 48KB， 12 way-associative ，每块64Bytes
AMD的ZEN3 L1D: 32KB, 8 way-associative ，每块64Bytes
但是不能一直增加相联度，否则会使时序变的很难看，所以有时候还是要增加set的个数和cacheline的大小，此时index+offset>12bit，这时候需要操作系统引入一个新功能：page color。具体的内容可以参考香山的Cache设计。
```

I will investigate `page color` later.