# Intel PMU
There are two kinds of profiling module: Counting and Sampling
Another way called tracing, it's not related to PMU

## Counting
In this mode, PMU just count how many times a event was occurred, not record the context info.
For example, it will make a statsitic about only how many insts or cycles were executed in a period.

How to use this mode:
`emon` or `perf stat` 

## Sample

In this mode, PMU will set a interrupt for CPU, when ref-cycle count achive a threshold it will send interrupt to APIC.
We could also set interrupt for all other PMUs (except ref-cycle), so in a group of PMU events we may receive interrupt when any kind of event achieve threshold. 
When core was handling this interrupt, it will not only store the count of each PMU, but also the PC (program count) and process id at that time, within these info we could post process them to filter module/library/function/inst in final result.
Given that we could set a interrupt for reference cycle event (physical time), its interval will be a physical time like 1ms (1000hz).
Some notes: 
a) If a core is in halt state, the time intervel may contain the halt state time (So the time may not be precise). 
b) Linux NMI_watchdog also use PMU event, when colletcing data, please disable it.
c) we will disable PMU profiling when core are running in interrupt handler in case it will bring noise to final result

How to use this mode:
`sep -ec ***`, `perf record` or `perf top`

## Tracing mode
This mode may not involves PMU. It seems to a pure software profling way.
It was also called instrument mode. 
It will use Tracepoints or kprobe or uprobe. when it hit this point, it will call handle and collect the context, but the cost is very expensive.

How to use this mode:
`perf record`, `perf trace`, `perf sched` or `perf timechart`

## LBR

LBR is a group of slots which store the recent `jump inst address` and `jump target address` (must be taken, if there is a jcc but not take, it will not be record), it seems to be 32 slots in current x64.

```
-------------------
| From 0  | To 0  |
| From 1  | To 1  |
...................
| From 31 | To 31 |
```
We could assume between `| To 0 |` to `| From 1 |`, there is no jump happend, so the insts inside them will be executed one by one, we could call it as a `super block`.
The difference between `super block` and `basic block` is that supper block may contain multiple `basic block`, because there may be some `jcc` inst inside it but not taken.
A note: there are some exception cases between `| To 0 |` to `| From 1 |`. Because there may happen context (process) switch, but LBR not aware of this, so there may be some case with a bigger `| To 0 |` than `| From 1 |`, we should fliter this data. (context switch may also cause ip change, why it didn't store in LBR? may check with Jonathan sharing later).


## Some useful command
`strace -F -e perf_event_open perf stat -a -e cycles -- sleep 1` could check on how these events were configured