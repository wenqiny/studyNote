# GPU insts
This file contains some weird GPU insts.

## LOP3
`LOP3` is an inst which used to do bit-wise computation among 3 elements.

```
lop3.b32 d, a, b, c, immLut;

Compute bitwise logical operation on inputs a, b, c and store the result in destination d.
```

For example:
`LOP3.LUT R42, R6, 0x7, RZ, 0xc0, !PT;` will do something like: `R42 = R6 & 0X7`, why?

The reason is that the `0xc0` in this inst define the behavior, the logical operation is defined by a look-up table which, for 3 inputs, can be represented as an 8-bit value specified by operand `immLut` as described below. `immLut` is an integer constant that can take values from 0 to 255, thereby allowing up to 256 distinct logical operations on inputs `a`, `b`, `c`.

```
ta = 0xF0;
tb = 0xCC;
tc = 0xAA;

immLut = F(ta, tb, tc);
```

Then

```
If F = (a & b & c);
immLut = 0xF0 & 0xCC & 0xAA = 0x80

If F = (a | b | c);
immLut = 0xF0 | 0xCC | 0xAA = 0xFE

If F = (a & b & ~c);
immLut = 0xF0 & 0xCC & (~0xAA) = 0x40

If F = ((a & b | c) ^ a);
immLut = (0xF0 & 0xCC | 0xAA) ^ 0xF0 = 0x1A
```

Therefore in the above case, `immLut = 0xc0 = 0xF0 & 0xCC`. so it's `R42 = R6 & 0X7`.

**QUESTION**: How could we use `ta = 0xF0; tb = 0xCC; tc = 0xAA;` to **surjective** all combination???

**BTW**, the `!PT` in this case ``LOP3.LUT R42, R6, 0x7, RZ, 0xc0, !PT;`` didn't means this inst shouldn't be executed, it's just an oprand, only something like `@PT` or `@!PT` at the prefix affect the inst be executed or not.