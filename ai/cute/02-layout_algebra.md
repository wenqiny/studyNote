## Layout algebra
There're serveral algebra we could apply between or inside `Layout`, which helps use to get a new `Layout`.

### Coalesce
Coalesce just like a `flatten`, but it will check whether the Layout is continuous or not, if it is, then we could coalesce it for all dims or some specific dims.

### Composition
Compostion means we use two or more `Layout` to compose a new `Layout`, like:
```
Functional composition, R := A o B
R(c) := (A o B)(c) := A(B(c))

Example
A = (6,2):(8,2)
B = (4,3):(3,1)

R( 0) = A(B( 0)) = A(B(0,0)) = A( 0) = A(0,0) =  0
R( 1) = A(B( 1)) = A(B(1,0)) = A( 3) = A(3,0) = 24
R( 2) = A(B( 2)) = A(B(2,0)) = A( 6) = A(0,1) =  2
R( 3) = A(B( 3)) = A(B(3,0)) = A( 9) = A(3,1) = 26
R( 4) = A(B( 4)) = A(B(0,1)) = A( 1) = A(1,0) =  8
R( 5) = A(B( 5)) = A(B(1,1)) = A( 4) = A(4,0) = 32
R( 6) = A(B( 6)) = A(B(2,1)) = A( 7) = A(1,1) = 10
R( 7) = A(B( 7)) = A(B(3,1)) = A(10) = A(4,1) = 34
R( 8) = A(B( 8)) = A(B(0,2)) = A( 2) = A(2,0) = 16
R( 9) = A(B( 9)) = A(B(1,2)) = A( 5) = A(5,0) = 40
R(10) = A(B(10)) = A(B(2,2)) = A( 8) = A(2,1) = 18
R(11) = A(B(11)) = A(B(3,2)) = A(11) = A(5,1) = 42
```

There're two majority observation:
1.  the function `R(c) = k` defined above can be written down as another `Layout R = ((2,2),3):((24,2),8)`.
2.  `B` and `R` is **compatible**, `compatible(B, R) = true`.
3.  The *codomain* of `B` should be a subset of *domain* of `A`.

That's easy to understand, because the *domain* for `B` and the newer `R` is the same domain, we could map the *domain* to *codomain* with two ways like:
1. `A(B(domain))`
2. `R(domain)`

Therefore what composition did is just compose the two `Layout` of `A` and `B`, but it didn't change the *domain*.

#### How to compute composition
We could divide-conquer for this question.

*  Divide `B` into serveral one-dim `Layout`: `B = (B_0, B_1, ...)`, for example, we could divide `B = (4,3):(3,1)` into `B = (4:3, 3:1)`.

* Then we could apply the compose for each dim in `B` with `A`: `A o B = A o (B_0, B_1, ...) = (A o B_0, A o B_1, ...)`, for example we could `A o B = ((6,2):(8,2) o 4:3, (6,2):(8,2) o 3:1)`.

Next, how could we compute something like `A o B_0`? we could divide this into two steps:
1. Elide some *domain* in `A` with stride of `B_0`, we deifine it as `d`, that's called a **divided-out**, because we could understand `B_0` will advanced by its stride, so which means if `B_0` advanced by `1`, the output address will advanced by `d`, that means some coordinate in `A` for this dim will not be mapped, so we should elide them, **at the same time, we should also multiply the stride**, because we elide some coordinate in this dim.
2. Keep the valid data in *domain* for `A` with shape of `B_0`, we define it as `s`, that's called a **modding out**, because the shape of `B_0` may be smaller than the shape of `A`, so we should to just keep the valid coordinate in this dim.

Let's use the above case to illustrate how the composition works:
```
A = (6,2):(8,2)
B = (4,3):(3,1) = (4:3, 3:1)
A o B = ((6,2):(8,2) o 4:3, (6,2):(8,2) o 3:1)
```

For `(6,2):(8,2) o 4:3`:
1. Elide *domain* with stride `3`:
   `(6,2):(8,2) / 3` -> `(2,2):(24,2)`, the fist dim of shape divide by `3`, so the first dim of stride should multiply by `3`.
2. Keep the *domain* with shape `4`:
   `(2,2):(24,2) % 4` -> `(2,2):(24,2)`, because `2*2` is equal to `4`, so we don't do anything for it.

The result is `(2,2):(24,2)`.

For `(6,2):(8,2) o 3:1`:
1. Elide *domain* with stride `1`:
   `(6,2):(8,2) / 1` -> `(6,2):(8,2)`.
2. Keep the *domain* with shape `3`:
   `(6,2):(8,2) / 3` -> `3:8`, because the shape of first dim is 6, which is greate than 3, so we just keep the first `3` elements in this dim.

The result is `3:8`

Then we could combin the two results `(2,2):(24,2)` and `3:8` together to `((2,2), 3):((24,2),8)`.


#### Case
There are two cases for composition:

```
// (12,(4,8)):(59,(13,1))
auto a = make_layout(make_shape (12,make_shape ( 4,8)),
                     make_stride(59,make_stride(13,1)));
// <3:4, 8:2>
auto tiler = make_tile(Layout<_3,_4>{},  // Apply 3:4 to mode-0
                       Layout<_8,_2>{}); // Apply 8:2 to mode-1

// (_3,(2,4)):(236,(26,1))
auto result = composition(a, tiler);
// Identical to
auto same_r = make_layout(composition(layout<0>(a), get<0>(tiler)),
                          composition(layout<1>(a), get<1>(tiler)));
```

We're computing `(12,(4,8)):(59,(13,1)) o <3:4, 8:2> = (_3,(2,4)):(236,(26,1))`, there is a figure:

![Composition 1](./pic/02-layout-algebra-composition-1.png)

We could see we just select serveral elements in `A` according to `B`, that's what composition do, it just map the *domain* in `B` to the *codomain* in `A`.

Another case:

```
// (12,(4,8)):(59,(13,1))
auto a = make_layout(make_shape (12,make_shape ( 4,8)),
                     make_stride(59,make_stride(13,1)));
// (3, 8)
auto tiler = make_shape(Int<3>{}, Int<8>{});
// Equivalent to <3:1, 8:1>
// auto tiler = make_tile(Layout<_3,_1>{},  // Apply 3:1 to mode-0
//                        Layout<_8,_1>{}); // Apply 8:1 to mode-1

// (_3,(4,2)):(59,(13,1))
auto result = composition(a, tiler);
```

It's `(12,(4,8)):(59,(13,1)) o <3, 8> = (_3,(2,4)):(236,(26,1))`, there is a figure:

![Composition 2](./pic/02-layout-algebra-composition-2.png)

It just select the 3x8 space at the left-top of `A` according to `B`, it's also a map between the *domain* of `B` to *codomain* of `A`.

### Complement
It's easy to understand what complement do, it just compute a new `Layout`, which means **how we repeat the original `Layout` to get the target `Layout`**.

* `complement(4:1, 24)` is `6:4`. Note that `(4,6):(1,4)` has cosize `24`. The layout `4:1` is effectively repeated 6 times with `6:4`.

* `complement(6:4, 24)` is `4:1`. Note that `(6,4):(4,1)` has cosize `24`. The "hole" in `6:4` is filled with `4:1`.

* `complement((4,6):(1,4), 24)` is `1:0`. Nothing needs to be appended.

* `complement(4:2, 24)` is `(2,3):(1,8)`. Note that `(4,(2,3)):(2,(1,8))` has cosize `24`. The "hole" in `4:2` is filled with `2:1` first, then everything is repeated 3 times with `3:8`.

* `complement((2,4):(1,6), 24)` is `3:2`. Note that `((2,4),3):((1,6),2)` has cosize `24` and produces unique indices.

* `complement((2,2):(1,6), 24)` is `(3,2):(2,12)`. Note that `((2,2),(3,2)):((1,6),(2,12))` has cosize `24` and produces unique indices.

From the above cases, we could compute complement in serveral steps:
1. Fill the "**Hole**", like for `complement(6:4, 24)`, the stride is `4`, which means there is a **hole** between each elements, we should add a `4:1` firstly to fill the **hole**.
2. Repeat to get the final size, like for `complement(4:2, 24)`, we will use `2:1` to fill the **hole** firstly, then we got a layout like `(4,(2):2,(1))`, the `size` of it is `4*2=8`, then we still need a new dim `24/8=3`, which stride is `8`, so the final output is `(2,3):(1,8)`.

### Division (Tiling)
We could define division with compostion, complement and concatenation:

$A \oslash B := A \circ (B,B^*)$

In cpp code it looks like:

```cpp
template <class LShape, class LStride,
          class TShape, class TStride>
auto logical_divide(Layout<LShape,LStride> const& layout,
                    Layout<TShape,TStride> const& tiler)
{
  return composition(layout, make_layout(tiler, complement(tiler, size(layout))));
}
```

#### How to understand division
We could think this question from the **unit perspective**, because for the `layout`, we could define its unit as $\frac{\text{index}}{\text{coordinate}}$ (the *coordinate* is not a tuple, but just an integer translated from the tuple), which means it's a map from *coordinate* to *index*, that's just what `Layout` do.

But for the `tiler`, it's actually a map between *coordinate* and *coordinate*, so its unit is $\frac{\text{coordinate}}{\text{coordinate}}$, therefore we could define the division as:

$\frac{\text{layout}}{\text{tiler}}=\frac{\frac{\text{index}}{\text{coordinate}}}{\frac{\text{coordinate}}{\text{coordinate}}}=\frac{\text{index}}{\text{coordinate}}$

After the division, the unit is still $\frac{\text{index}}{\text{coordinate}}$.

#### Case study
We could see two cases: 1-D and 2-D.

##### 1-D case
Consider tiling the 1-D layout `A = (4,2,3):(2,1,8)` with the tiler `B = 4:2`, we could do computation in next steps:

```
A = (4,2,3):(2,1,8)
B = 4:2

B* = (2,3):(1,8)
(B,B*) = (4,(2,3)):(2,(1,8))
A o (B,B*) = (A o B, A o B*, ...)
           = ((4,2,3):(2,1,8) o 4:2, (4,2,3):(2,1,8) o 2:1, (4,2,3):(2,1,8) o 3:8)
           = ((2,2):(4,1), 2:2, 3:8)
           = ((2,2),(2,3):(4,1),(2,8))
```

We could illustrate it as:
![Division 1-D](./pic/02-layout-algebra-division-1d.png)

In the figure, we could see the gray elements is the `tiler`, their *coordinate* is `B = 4:2`, which just map a *coordinate* to a new *coordinate* in `4` elements with stride as `2`.

Then the division will do:
1. Compact all the elements in the gray as together as the first cluster.
2. Repeat the cluster multiple times to get the same shape as `A`.
   
##### 2-D case
Consider a 2-D layout `A = (9,(4,8)):(59,(13,1))` and want to apply `3:3` down the columns (mode-0) and `(2,4):(1,8)` across the rows (mode-1). This means the tiler can be written as `B = <3:3, (2,4):(1,8)>`.

We could compute it in these steps:
```
A = (9,(4,8)):(59,(13,1))
B = <3:3,(2,4):(1,8)>

We could compute this in both row and column dims and then split A/B to A/Br and A/Bc

Ar = (4,8):(13,1)
Ac = 9:59

Br = (2,4):(1,8)
Bc = 3:3

Br* = 4:2
Bc* = 3:1

Ar o (Br,Br*) = (Ar o Br, Ar o Br*)
              = ((4,8):(13,1) o (2,4):(1,8), (4,8):(13,1) o 4:2)
              = ((4,8):(13,1) o 2:1, (4,8):(13,1) o 4:8, (4,8):(13,1) o 4:2)
              = (2:13, 4:2, (2,2):(26,1))
              = ((2,4),(2,2):(13,2),(26,1))

Ac o (Bc,Bc*) = (Ac o Bc, Ac o Bc*)
              = (9:59 o 3:3, 9:59 o 3:1)
              = (3:177, 3:59)
              = ((3,3):(177,59))

Combain them together we got:
((3,3),(2,4),(2,2):(177,59),(13,2),(26,1))
```

We could illustrate it as:
![Division 2-D](./pic/02-layout-algebra-division-2d.png)

We could also see it do two steps:
1. Compact gray elements
2. Repeat the compacted cluster.

### Product
Just like division we could define product with some basic algebra:

$A \otimes B := (A, A^* \circ B)$

and implemented in CuTe as
```cpp
template <class LShape, class LStride,
          class TShape, class TStride>
auto logical_product(Layout<LShape,LStride> const& layout,
                     Layout<TShape,TStride> const& tiler)
{
  return make_layout(layout, composition(complement(layout, size(layout)*cosize(tiler)), tiler));
}
```

From the above formula, we could understand the product in two steps:
1. Take the `A` as a cpoy to as the base.
2. Repeat `A` the composition of `A*` to `B` times (`A*` at here is the complement of the `size(A)*cosize(B)`, because we want to repeat `size(A)` for `cosize(B)` times).

#### Case study
Just like division, we use two cases:

##### 1-D case
Consider the 1-D layout `A = (2,2):(4,1)` according to `B = 6:1`. Informally, this means that we have a 1-D layout of 4 elements defined by A and we want to reproduce it 6 times.

```
A = (2,2):(4,1)
B = 6:1

size(A)*cosize(B)=4*6=24
A* = (2,3):(2,8)

A* o B = ((2,3):(2,8) o 6:1)
       = ((2,3):(2,8))

Concatenate A and A* o B:
((2,2),(2,3):(4,1),(2,8))
```

We could illustrate it as:
![Product 1-D](./pic/02-layout-algebra-product-1d.png)

We could see it do it in two steps:
1. Copy `A` into the gray element in `B`.
2. Repeat the copy for all the element in `B`.

##### 2-D case
Consider the 2-D layout `A = (2,5):(5,1)` and `B = (3,4):(1,3)`.

We could compute simlar as 2-D case for division:
```
A = (2,5):(5,1)
B = (3,4):(1,3)


size(A)*cosize(B)=10*(2*1+3*3+1)=120
A* = (12:10)

A* o B = (12:10) o (3,4):(1,3)
       = ((12:10) o (3,1), (12:10) o (4,3))
       = (3:10, 4:30)

Combain them together we got:
((2,5),(3,4):(5,1),(10,30))

And we could also combain some dims to write it also
(6,(5,4):5,(1,30))
```

Let's illustrate it in the below figure:
![Product 2-D](./pic/02-layout-algebra-product-2d.png)

**TODO: Not very clear about the blocked_product and raked_product, and how to do product in x and y dims, try to review later.**

We could see it just repeat the gray parts by 3x4 times.

### Reference
[Official doc](https://github.com/NVIDIA/cutlass/blob/v4.0.0/media/docs/cpp/cute/02_layout_algebra.md)
