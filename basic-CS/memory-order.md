# Memory order
Memory order is an interesting topic in programing, sometime the **memory reorder** may happen, there're two causes:
1. software (compiler)
2. hardware (OOO or other hardware feature)

A [vedio](https://www.bilibili.com/video/BV16C4y1e7aP) about it.

## Software reorder
Let's image such a case:
```cpp
int a, b;

int func();

void test() {
    a = func();
    b = 2;
}
```

The asm code may looks like:
```
test():
    sub     rsp, 8
    call    func() <------------------- calling function
    mov     DWORD PTR b[rip], 2 <------ store b
    mov     DWORD PTR a[rip], eax <---- store function results to a
    add     rsp, 8
    ret
```

We could see the compiler sometimes optimize it and make **store b**, before **store a**, what we could do is adding a memory fence, like:
```cpp
int a, b;

int func();

void test() {
    a = func();
    asm("":::"memory");
    b = 2;
}
```

After adding such a fence, the compiler will not do this optimization.


## Hardware reorder
In modern CPU, it uses OOO to execute insts, so the order of each memory opearation is not always same as the define in the asm code, let's see the below case:

```cpp
//g++ mfence.cpp --std=c++20 -lpthread
//g++ mfence.cpp --std=c++20 -lpthread -O3
#include <stdio.h>
#include <semaphore.h>
#include <thread>

int v1, v2, r1, r2;
sem_t start1, start2, complete;

void thread1()
{
    while(true)
    {
        sem_wait(&start1);  //wait for start
        v1 = 1;
        
        //asm ("mfence" ::: "memory");

        r1 = v2;
        sem_post(&complete);  //complete & trigger a signal
    }
}

void thread2()
{
    while(true)
    {
        sem_wait(&start2);  //wait for start
        v2 = 1;
        
        //asm ("mfence" ::: "memory");

        r2 = v1;
        sem_post(&complete);  //complete & trigger a signal
    }
}

int main()
{
    sem_init(&start1, 0, 0);
    sem_init(&start2, 0, 0);
    sem_init(&complete, 0, 0);

    std::thread t1(thread1);
    std::thread t2(thread2);

    for(int i = 0; i < 300000; i++)
    {
        v1 = v2 = 0;
        
        sem_post(&start1);  //start t1
        sem_post(&start2);  //start t2

        //wait for t1&t2 completion
        sem_wait(&complete);
        sem_wait(&complete);

        if((r1 == 0) && (r2 == 0))
        {
            printf("reorder detected @ %d\n", i);
        }
    }

    t1.detach();
    t2.detach();
}
```

We could compare and run it, on my local x64 machine, we could see there is some case that **both r1 and r2 is 0**, that's means the CPU execute the later inst before pervious inst, why?

The asm code looks like:
```
thread1():
    push    r14
    push    rbx
    push    rax
    lea     rbx, [rip + start1]
    lea     r14, [rip + complete]
.LBB0_1:
    mov     rdi, rbx
    call    sem_wait@PLT
    mov     dword ptr [rip + v1], 1 <------- store 1 to v1
    mov     eax, dword ptr [rip + v2] <----- load v2
    mov     dword ptr [rip + r1], eax <----- store v2 to r1
    mov     rdi, r14
    call    sem_post@PLT
    jmp     .LBB0_1
```

In the x64 CPU, it don't allow load-load, and store-store reorder, which means **all the store before a store inst will be retired before it**, and same as load.

But there is an exception is the store-load reordering, because there is a hardware feature called **write buffer**, which means some writer results will not be store in cache or memory immediatly (due to store latency), but store in **write buffer**, so it's not visiable to other CPU cores, which may make other CPUs see the CPU do the later load but not the previous store (because the store results was not in cache/memory, other CPUs couldn't see **write buffer**).

On x64, the culprit is the **writer buffer**, but for other CPU arch, the memory order is more weak, it means there is more case for they to do memory reordering.

As the solution for the reodering is to add a memory fence, like `asm ("mfence" ::: "memory");`, or using the `atomic<int>` for v1 and v2, because it have the semantic to make its results to be explict visiable to other CPUs. 