## Lecture 03
1. uArch of nvidia gpus: a rtx3090 consists 82 SMs, each SM consists 4 partition, each partition consists 32 FP32 cores (16 FP32 and 16 INT32).
2. Occupancy GPU as much as possible.
3. Avoid thread divergence (control flow).
4. Not read/write too much from long distance memery (an optimization case for matrix multiply by share memory).