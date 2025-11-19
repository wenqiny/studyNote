## MMA atom
This chapter is about the MMA atom, which is the basic unit of how we use tensor core to do mma.

### Contents
MMA in CUTE contains two parts: `Operation` and `MMA_Traits`.

#### Operation structs
Operation is something like: `SM70_8x8x4_F32F16F16F32_NT`.

* "SM70" refers to Volta.

* "8x8x4" refers to M = 8, N = 8, and K = 4,
  the dimensions of the MMA operation that the quadpair performs
  (see below). This is reflected in the PTX as `.m8n8k4.`.

* "F32F16F16F32" refers to the element types
  of the four matrix operands A, B, C, and D.
  An MMA computes D = C + A * B,
  so we read the types from left to right:
  D is F32 (`float`), A is F16 (half),
  B is F16 (half), and C is F32 (`float`). This is reflected in the PTX instruction name as `.f32.f16.f16.f32`.

* "NT" means that the PTX instruction is designed for inputs A as M-major (not transposed, column-major)
  and inputs B as N-major (transposed, row-major). This is reflected in the PTX instruction name as `.col.row.`.

This struct just contains some data about the registers, because we could understand this `Operation` just care about the ptx inst's self, so it just have:

```
using DRegisters = float[8];
using ARegisters = uint32_t[2];
using BRegisters = uint32_t[2];
using CRegisters = float[8];
```


#### Traits
Traits have more higher level contents than the `Operation`, like:

* `ValTypeD`: Logical compute type of the D matrix

* `ValTypeA`: Logical compute type of the A matrix

* `ValTypeB`: Logical compute type of the B matrix

* `ValTypeC`: Logical compute type of the C matrix

* `Shape_MNK`: Logical MxNxK shape of the MMA operation

* `ThrID`: Logical thread mapping within the single MMA operation
  (specifying the thread, quadpair, warp, or warpgroup view)

* `ALayout`: Mapping of (thread,value) pairs to coordinates in the MxK A matrix

* `BLayout`: Mapping of (thread,value) pairs to coordinates in the NxK B matrix

* `CLayout`: Mapping of (thread,value) pairs to coordinates in the MxN C matrix

It cares more about how the data for the whole MMA in lower level storage, like the data type and the `Layout`.