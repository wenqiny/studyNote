## Tensor
This chapter is about `Tensor`, it comprises two parts: `Layout` and `Engine`.

The `Layout` is what we discussed in previous chapter, it maps a *coordinate* to an *index*.

The `Engine` is a random-access iterator, which may contain some data like base pointer, data type and etc.

### Tensor creation
There are two types of `Tensor`
1. Owning `Tensor`, it means the `Tensor` take the ownership of the address space which it point to. When you copy the Tensor, you (deep-)copy its elements, and the Tensor's destructor deallocates the array of elements.
2. Nonowning `Tensor`, it just take the base pointer of the address space as its parameter. Copying the Tensor doesn't copy the elements, and destroying the Tensor doesn't deallocate the array of elements.

What we usually used is the Nonowning `Tensor` in CUTE.

### Cases
#### Owning Tensor
For Owning `Tensor`, we don't need to explictly pass an address to space to it, just like:

```
// Register memory (static layouts only)
Tensor rmem_4x8_col = make_tensor<float>(Shape<_4,_8>{});
Tensor rmem_4x8_row = make_tensor<float>(Shape<_4,_8>{},
                                         LayoutRight{});
Tensor rmem_4x8_pad = make_tensor<float>(Shape <_4, _8>{},
                                         Stride<_32,_2>{});
Tensor rmem_4x8_like = make_tensor_like(rmem_4x8_pad);
```

We just need to pass the data type and `Layout`, then CUTE will help us create the `Tensor`.

If we print them:

```
rmem_4x8_col  : ptr[32b](0x7fff48929460) o (_4,_8):(_1,_4)
rmem_4x8_row  : ptr[32b](0x7fff489294e0) o (_4,_8):(_8,_1)
rmem_4x8_pad  : ptr[32b](0x7fff489295e0) o (_4,_8):(_32,_2)
rmem_4x8_like : ptr[32b](0x7fff48929560) o (_4,_8):(_8,_1)
```

We could see all the tensors have **different** address.

#### Nonowning Tensor

For Nonowning `Tensor`, we should explictly pass an address/point to it, just like:

```
float* A = ...;

// Untagged pointers
Tensor tensor_8   = make_tensor(A, make_layout(Int<8>{}));  // Construct with Layout
Tensor tensor_8s  = make_tensor(A, Int<8>{});               // Construct with Shape
Tensor tensor_8d2 = make_tensor(A, 8, 2);                   // Construct with Shape and Stride

// Global memory (static or dynamic layouts)
Tensor gmem_8s     = make_tensor(make_gmem_ptr(A), Int<8>{});
Tensor gmem_8d     = make_tensor(make_gmem_ptr(A), 8);
Tensor gmem_8sx16d = make_tensor(make_gmem_ptr(A), make_shape(Int<8>{},16));
Tensor gmem_8dx16s = make_tensor(make_gmem_ptr(A), make_shape (      8  ,Int<16>{}),
                                                   make_stride(Int<16>{},Int< 1>{}));

// Shared memory (static or dynamic layouts)
Layout smem_layout = make_layout(make_shape(Int<4>{},Int<8>{}));
__shared__ float smem[decltype(cosize(smem_layout))::value];   // (static-only allocation)
Tensor smem_4x8_col = make_tensor(make_smem_ptr(smem), smem_layout);
Tensor smem_4x8_row = make_tensor(make_smem_ptr(smem), shape(smem_layout), LayoutRight{});
```

We could print them:

```
tensor_8     : ptr[32b](0x7f42efc00000) o _8:_1
tensor_8s    : ptr[32b](0x7f42efc00000) o _8:_1
tensor_8d2   : ptr[32b](0x7f42efc00000) o 8:2
gmem_8s      : gmem_ptr[32b](0x7f42efc00000) o _8:_1
gmem_8d      : gmem_ptr[32b](0x7f42efc00000) o 8:_1
gmem_8sx16d  : gmem_ptr[32b](0x7f42efc00000) o (_8,16):(_1,_8)
gmem_8dx16s  : gmem_ptr[32b](0x7f42efc00000) o (8,_16):(_16,_1)
smem_4x8_col : smem_ptr[32b](0x7f4316000000) o (_4,_8):(_1,_4)
smem_4x8_row : smem_ptr[32b](0x7f4316000000) o (_4,_8):(_8,_1)
```

It seems all of them share a same address.


### Tiling a Tensor

We could use similar method for `Layout` to tile a `Tensor`, just like:

```
   composition(Tensor, Tiler)
logical_divide(Tensor, Tiler)
 zipped_divide(Tensor, Tiler)
  tiled_divide(Tensor, Tiler)
   flat_divide(Tensor, Tiler)
```

### Slicing a Tensor
We could slice a Tensor with `_` (an instance of `cute::Underscore`， it just same as `:` in `pytorch`) and some specific number for *coordinate*, let's see some examples:

```
// ((_3,2),(2,_5,_2)):((4,1),(_2,13,100))
Tensor A = make_tensor(ptr, make_shape (make_shape (Int<3>{},2), make_shape (       2,Int<5>{},Int<2>{})),
                            make_stride(make_stride(       4,1), make_stride(Int<2>{},      13,     100)));

// ((2,_5,_2)):((_2,13,100))
Tensor B = A(2,_);

// ((_3,_2)):((4,1))
Tensor C = A(_,5);

// (_3,2):(4,1)
Tensor D = A(make_coord(_,_),5);

// (_3,_5):(4,13)
Tensor E = A(make_coord(_,1),make_coord(0,_,1));

// (2,2,_2):(1,_2,100)
Tensor F = A(make_coord(2,_),make_coord(_,3,_));
```

We could illustrate them as:

![Slicing a Tensor](./pic/03-tensor-slicing.png)

From the above figure we have two observations:
1. The slicing will also change the base pointer for the `Tensor`, like for the first case `B=A(2,_)`, we could see it move the base pointer from `0` to `8` (first elements in gray).
2. Dimesion for with coordinate as specific value will be keep, with `_` will keep all coordinate in this dim.

### Partitioning a Tensor

What partition do is that it first use a `tiler` to divide a `Tensor`, then we use some *coordinate* (usually be the blockIdx or threadIdx) to slice it.

Let's see a case:

```
Tensor A = make_tensor(ptr, make_shape(8,24));  // (8,24)
auto tiler = Shape<_4,_8>{};                    // (_4,_8)

Tensor tiled_a = zipped_divide(A, tiler);       // ((_4,_8),(2,3))
```
#### Inner partition
After the division, if we want to ask each block to process a `4x8` partition, we could do:

```
Tensor cta_a = tiled_a(make_coord(_,_), make_coord(blockIdx.x, blockIdx.y));  // (_4,_8)
```

We call this an *inner-partition* because it keeps the inner "tile" mode. This pattern of applying a tiler and then slicing out that tile by indexing into the remainder mode is common and has been wrapped into its own function `inner_partition(Tensor, Tiler, Coord)`.

There is another wrapper called `local_tile(Tensor, Tiler, Coord)` which is just another name for inner_partition. The `local_tile` partitioner is very often applied at the threadgroup level to partition tensors into tiles across threadgroups.

#### Outer partition
We could also use another way to do the partition for outer dims:

```
Tensor thr_a = tiled_a(threadIdx.x, make_coord(_,_)); // (2,3)
```

We call this an *outer-partition* because it keeps the outer "rest" mode. This pattern of applying a tiler and then slicing into that tile by indexing into the tile mode is common and has been wrapped into its own function `outer_partition(Tensor, Tiler, Coord)`.

Another wrapper `local_partition(Tensor, Layout, Idx)`, which is a rank-sensitive wrapper around outer_partition that transforms the `Idx` into a *Coord* using the inverse of the `Layout` and then constructs a `Tiler` with the same top-level shape of the `Layout`.

**TODO: try to understand how local_partition works and diff between it and local_tile**.

#### Thread-Value partitioning
Another partition is thread-value (TV) partition, which means we could use a `Layout` for each thread to get the data from a `Tensor`, that's implement by composition:

```
// Construct a TV-layout that maps 8 thread indices and 4 value indices
//   to 1D coordinates within a 4x8 tensor
// (T8,V4) -> (M4,N8)
auto tv_layout = Layout<Shape <Shape <_2,_4>,Shape <_2, _2>>,
                        Stride<Stride<_8,_1>,Stride<_4,_16>>>{}; // (2,4),(2,2):(8,1):(4,16)

// Construct a 4x8 tensor with any layout
Tensor A = make_tensor<float>(Shape<_4,_8>{}, LayoutRight{});    // (4,8):(8,1)
// Compose A with the tv_layout to transform its shape and order
Tensor tv = composition(A, tv_layout);                           // ((2,4),(2,2):(2,8),(1,4))
// Slice so each thread has 4 values in the shape and order that the tv_layout prescribes
Tensor  v = tv(threadIdx.x, _);                                  // (2,2):(1,4)
```

We could illustrate it as (**NOTE: the number in the TV layout image is not the *index*, but the *coordinate* of the *domain* for the `Tensor`**):

![TV partition](./pic/03-tensor-TV-partition.png)

We could see the finnal TV partition as `((2,4),(2,2):(2,8),(1,4))`.

### Reference
[Official doc](https://github.com/NVIDIA/cutlass/blob/v4.0.0/media/docs/cpp/cute/03_tensor.md)