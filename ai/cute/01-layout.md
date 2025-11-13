## Layout
`Layout` is is a key concept in CUTE, it comprises `Shape` and `Stride`, both of them are `Tuple`.

### Method
Layout have serveral methods, like `rank`, `depth`, `size` and etc.

Most of them are easy to understand, what I want to mention at here is the diff between `rank` and `depth`:
1. `rank` gets how many ranks for the `Layout` or `Shape` without nesting dimesions, like `(1, 2, 3)` is **3 ranks**, but for `(1, (2, 3))`, there're just **2 ranks**.
2. `depth` gets how many depth for the `Layout` or `Shape`, like for `1`, the depth is **0**, for `(1, 2, 3)` the depth is **1**, for `(1, (2, 3))`, the depth is **2**.


### How layout works
What `Layout` do is to **map coordinate in `Shape` space to index in `Stride` space**, it's easy to understand for a non-nesting case, but we could see a case with nesting shapes.

For the case:
```
Layout s2xh4 = make_layout(make_shape (2,make_shape (2,2)),
                           make_stride(4,make_stride(2,1)));
Layout s2xh4_col = make_layout(shape(s2xh4),
                               LayoutLeft{});
```

If we print them, it shows:
```
s2xh4     :  (2,(2,2)):(4,(2,1))
s2xh4_col :  (2,(2,2)):(_1,(2,4))
```

It's easy to understand the number in the shape and stride, but how they co-work to do the mapping? We could do a experiment:

```c++
template <class Shape, class Stride>
void print2D(Layout<Shape,Stride> const& layout)
{
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m,n));
    }
    printf("\n");
  }
}
```

which produces the following output for the above examples.

```
> print2D(s2xh4)
  0    2    1    3
  4    6    5    7
> print2D(s2xh4_col)
  0    2    4    6
  1    3    5    7
```

The first dim of `2` is easy to understand, it's the row dim.

For this case the `int n = 0; n < size<1>(layout); ++n` loop will iterate among `(2,2)` in the col dim in the `Shape` of `(2,(2,2))`, the question came here, **how to may `(2, 2)` to an index in `range(0, 4)`**?

The answer is: **it's same as how we calculate the `threadIdx` in CUDA with `threadIdx.x` and `threadIdx.y` and `threadIdx.z`**, which means the `2` elements in the first dim of `2` is the `0th` and `1st` elements, then the `2` elements in the second dim of `2` is the `2nd` and `3rd` elements.

We could also use `print_layout` to directly get the layout map (but it just suppost 2d case, which means the `rank` of the `Shape` is `2`):
```
> print_layout(s2xh4)
(2,(2,2)):(4,(2,1))
      0   1   2   3
    +---+---+---+---+
 0  | 0 | 2 | 1 | 3 |
    +---+---+---+---+
 1  | 4 | 6 | 5 | 7 |
    +---+---+---+---+
```

### Matrix layout
For a matrix `Layout` is just same as above 2d case, but if we define the row in a nesting case:
```
Shape:  ((2,2),2)
Stride: ((4,1),2)
  0   2
  4   6
  1   3
  5   7
```
It's also easy to understand.


### Coordinate Mapping
The map from an input coordinate to a natural coordinate is the application of a colexicographical order (reading right to left, instead of "lexicographical," which reads left to right) within the `Shape`.

Take the shape `(3,(2,3))`, for example. This shape has three coordinate sets: the 1-D coordinates, the 2-D coordinates, and the natural (h-D) coordinates.

|  1-D  |   2-D   |   Natural   | |  1-D  |   2-D   |       Natural   |
| ----- | ------- | ----------- |-| ----- | ------- | ----------- |
|  `0`  | `(0,0)` | `(0,(0,0))` | |  `9`  | `(0,3)` | `(0,(1,1))` |
|  `1`  | `(1,0)` | `(1,(0,0))` | | `10`  | `(1,3)` | `(1,(1,1))` |
|  `2`  | `(2,0)` | `(2,(0,0))` | | `11`  | `(2,3)` | `(2,(1,1))` |
|  `3`  | `(0,1)` | `(0,(1,0))` | | `12`  | `(0,4)` | `(0,(0,2))` |
|  `4`  | `(1,1)` | `(1,(1,0))` | | `13`  | `(1,4)` | `(1,(0,2))` |
|  `5`  | `(2,1)` | `(2,(1,0))` | | `14`  | `(2,4)` | `(2,(0,2))` |
|  `6`  | `(0,2)` | `(0,(0,1))` | | `15`  | `(0,5)` | `(0,(1,2))` |
|  `7`  | `(1,2)` | `(1,(0,1))` | | `16`  | `(1,5)` | `(1,(1,2))` |
|  `8`  | `(2,2)` | `(2,(0,1))` | | `17`  | `(2,5)` | `(2,(1,2))` |

Each coordinate into the shape `(3,(2,3))` has two *equivalent* coordinates and all equivalent coordinates map to the same natural coordinate. To emphasize again, because all of the above coordinates are valid inputs, a Layout with Shape `(3,(2,3))` can be used as if it is a 1-D array of 18 elements by using the 1-D coordinates, a 2-D matrix of 3x6 `(3, 6)` elements by using the 2-D coordinates, or a h-D tensor of 3x(2x3) elements by using the h-D (natural) coordinates.

### idx2crd

There is also a method `idx2crd` which could help us to map an `index` to a `coordinate` in a `Sahpe`.
```
auto shape = Shape<_3,Shape<_2,_3>>{};
print(idx2crd(   16, shape));                                // (1,(1,2))
print(idx2crd(_16{}, shape));                                // (_1,(_1,_2))
print(idx2crd(make_coord(   1,5), shape));                   // (1,(1,2))
print(idx2crd(make_coord(_1{},5), shape));                   // (_1,(1,2))
print(idx2crd(make_coord(   1,make_coord(1,   2)), shape));  // (1,(1,2))
print(idx2crd(make_coord(_1{},make_coord(1,_2{})), shape));  // (_1,(1,_2))
```

### crd2idx
There is also `crd2idxx` to map `coordinate` to `index` with a `Shape`, it could take all compatible `coordinate` like `1-d`, `2-d` or `h-d` (natural).

```
auto shape  = Shape <_3,Shape<  _2,_3>>{};
auto stride = Stride<_3,Stride<_12,_1>>{};
print(crd2idx(   16, shape, stride));       // 17
print(crd2idx(_16{}, shape, stride));       // _17
print(crd2idx(make_coord(   1,   5), shape, stride));  // 17
print(crd2idx(make_coord(_1{},   5), shape, stride));  // 17
print(crd2idx(make_coord(_1{},_5{}), shape, stride));  // _17
print(crd2idx(make_coord(   1,make_coord(   1,   2)), shape, stride));  // 17
print(crd2idx(make_coord(_1{},make_coord(_1{},_2{})), shape, stride));  // _17
```

### Summary

* The `Shape` of a `Layout` defines its coordinate space(s).

    * Every `Layout` has a 1-D coordinate space.
      This can be used to iterate over the coordinate spaces in a colexicographical order.

    * Every `Layout` has a R-D coordinate space,
      where R is the rank of the layout.
      The colexicographical enumeration of the R-D coordinates
      correspond to the 1-D coordinates above.

    * Every `Layout` has an h-D (natural) coordinate space where h is "hierarchical." These are ordered colexicographically and the enumeration of that order corresponds to the 1-D coordinates above. A natural coordinate is *congruent* to the `Shape` so that each element of the coordinate has a corresponding element of the `Shape`.

* The `Stride` of a `Layout` maps coordinates to indices.

    * The inner product of the elements of the natural coordinate with the elements of the `Stride` produces the resulting index.

For each `Layout` there exists an integral `Shape` that is that compatible with that `Layout`. Namely, that integral shape is `size(layout)`. We can then observe that

> Layouts are functions from integers to integers.

### Rerference
[Offcial markdown](https://github.com/NVIDIA/cutlass/blob/v4.0.0/media/docs/cpp/cute/01_layout.md)