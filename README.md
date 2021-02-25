```reentrant and thread-safe

reentrant and thread-safe是无关的，虽然在绝大多数情况下可以看作reentrant是thread-safe的充分不必要条件，但是有特例：
int t;

void swap(int *x, int *y)
{
int s;

s = t; // save global variable
t = *x;
*x = *y;
// hardware interrupt might invoke isr() here!
*y = t;
t = s; // restore global variable
}


这是一个reentrant but not thread-safe, 在未加锁使用了全局变量t的情况下显然not thread-safe。

但是这个函数却是reentrant的因为他虽然访问和修改了全局变量t，但是它在函数结束时却保证了全局变量t一定不会修改。
因为在单线程内重入的话，一定要先完成调用才会返回上一个调用，这样在完成任意调用后都不修改全局变量t则不会影响整个调用栈，由数学归纳法可证。（但是acquire lock并不可，会造成dead lock）
但是在多线程内不可能保证一定会完成一个调用在执行另一个调用，它们的执行情况是完全随机的，这就是单线程可重入和多线程的最主要区别。
就像printf之所以thread-safe but not reentrant （假设锁住stdout）是因为会切换，即使拿不到锁也会切回已经拿到锁的线程执行，但是单线程的重入并不能切回，因为他必须先完成目前的调用。
