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

### Complement
It's easy to understand what complement do, it just compute a new `Layout`, which means **how we repeat the original `Layout` to achive the target `Layout`**.

* `complement(4:1, 24)` is `6:4`. Note that `(4,6):(1,4)` has cosize `24`. The layout `4:1` is effectively repeated 6 times with `6:4`.

* `complement(6:4, 24)` is `4:1`. Note that `(6,4):(4,1)` has cosize `24`. The "hole" in `6:4` is filled with `4:1`.

* `complement((4,6):(1,4), 24)` is `1:0`. Nothing needs to be appended.

* `complement(4:2, 24)` is `(2,3):(1,8)`. Note that `(4,(2,3)):(2,(1,8))` has cosize `24`. The "hole" in `4:2` is filled with `2:1` first, then everything is repeated 3 times with `3:8`.

* `complement((2,4):(1,6), 24)` is `3:2`. Note that `((2,4),3):((1,6),2)` has cosize `24` and produces unique indices.

* `complement((2,2):(1,6), 24)` is `(3,2):(2,12)`. Note that `((2,2),(3,2)):((1,6),(2,12))` has cosize `24` and produces unique indices.

### Division (Tiling)

