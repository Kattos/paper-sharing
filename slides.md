---
theme: default
class: "text-center"
highlighter: shiki
---

## DiffTaichi: Differentiable Programming for Physical Simulation

<br>

### Yuanming Hu (and many others)

<br>

### MIT CSAIL

---

# Background

<hr color="lightgray">

<p>

There exists some approachs to implementing differntiable physical simulators:

* TensorFlow (or other frameworks designed for deep learning) leads to poor performance
* Hand-writing CUDA kernels requires a lot of time

<img src="/images/productivity-performance.png" class="mx-auto w-lg" />

</p>

<!--

* 过去的工具在物理模拟微分领域难以同达到高生产力和高性能
    * tf 是为神经网络设计的，而神经网络与物理模拟有许多不同
        * 物理模拟中算子数量级大于神经网络
        * 神经网络中的高级算子 (conv 等) 在物理模拟中不能进一步定制，用户需要用低级 op 组成需要的高级 op
    * cuda 没有自动微分，需要手动编写大量的微分 kernel
    
-->

---

# Taichi to DiffTaichi

<hr color="lightgray">

<p>

* **[Taichi](https://www.taichi-lang.org/)** is an imperative programming language that delivers both high performance and high productivity

<img src="/images/taichi.png" class="mx-auto w-sm my-2rem" />

* Based on Taichi, the authors present **DiffTaichi** with special considerations:
    * **Megakernels**: fuse multiple stage of computation into a single kernel
    * **Imperative Parallel Programming**: simplify tasks with parallel loops and control flows
    * **Flexible Indexing**: manipulate array elements via arbitrary indexing

</p>

<!--

* taichi: 作者团队之前的工作，嵌入在 python 中的高性能 dsl，一些算法使用 taichi 比原生 python 性能提升可以达到百倍 
    * ([用 Taichi 加速 Python：提速 100+ 倍！](https://zhuanlan.zhihu.com/p/547123604))
* 为了支持 AD，作者在 taichi 的基础上进行改进提出了 diff taichi，与其他的 AD 工具/框架相比， diff taichi 主要关注简化物理模拟编写工作的几个方面
    * 巨型 kernel 允许用户将多个阶段的计算融合到一个 kernel 中，随后使用 SCT 和 JIT 对其进行微分，相比 tf 和 pytorch 对于物理模拟更有效率
    * 大多数物理模拟器是使用 fortran/c++ 等命令式语言编写而不是深度学习中流行的声明式语言，因此 diff taichi 提供了命令式编程以便现有代码的移植
    * 物理模拟中很多操作不是 elementwise 的，diff taichi 为用户提供了更灵活的 index 访问用于物理模拟的开发和维护，明确的 index 语法也更容易进行访存优化

-->

---

# Automatic Differentiation in DiffTaichi

<hr color="lightgray">

<p>

Comparison of AD system design:

* Source Code Transformation (SCT): high performance yet poor flexibility
* Tracing: high flexibility yet poor performance

To get both performance and flexibilty, **DiffTaichi**'s AD system is two-scale:

<img src="/images/two-scales.png" class="mx-auto w-xl" />

* Within kernels <span class="text-red-500">(red box)</span>: use SCT
* Outside of kernels <span class="text-green-500">(green box)</span>: use a light-weight tape for end-to-end differentiation

</p>

<!--

* diff taichi 主要考虑两种实现方式
    1. 源代码转换：将 primal 代码以某种方式转化为 adjoint 代码，这种方式具有较高的性能但灵活性较差
    2. tracing: 使用额外的数据结构记录梯度以便后续使用，这种方式灵活性高但性能较差
* 为了同时获得性能和灵活的优势，diff taichi 使用在不同的范围内使用不用的 AD 方式
    * 对于 kernel 内的 AD 以源代码转换的方式实现
    * 对于 kernel 之间端到端的 AD 利用一个轻量级的 tape 数据结构以 tracing 的方式实现
* 图中白色部分是复用 diff 的工作，蓝色部分是 diff taichi 的创新
    * 红框中 python 前端和 tape 系统是端到端 AD 部分
    * 绿框中 ast 预处理和生成伴随是 kernel 内的 AD 部分

-->

---

# Assumption

<hr color="lightgray">

<br>

<p>

The goal of AD is clear:

<div class="my-5rem">

$$
f(X) = Y \xrightarrow{\text{AD}} f^{*}(X, Y^{*}) = X^{*}
$$

</div>

Some assumptions on imperative kernels are made to avoid overwriting each other's output:

1. If a global tensor element is written more than once, then starting from the second write, the write must come **in the form of an atomic add ("accumulation")**
2. No read access happen to a global tensor element **until its accumulation is done**

</p>

<!--

* 为了生成正确的伴随函数，需要保证运算的输出不会被意外地覆盖，因此 diff taichi 对系统的读写做出了一些限制
    * 对于写操作，如果一个元素将被写多次则必须以原子的方式进行（多次更新/累加梯度）
    * 对于读操作，必须在所有的原子写操作完成之后才能开始进行

-->

---

# Local AD: Preprocessing

<hr color="lightgray">

<p>

To make later AD easier, two code transforms are introduced to simplify loop body in Taichi kernel:

1. Flatten branching: replace `if` statements with ternary operators `select`

<div grid="~ cols-2 gap-10">

```cpp
int a = 0;
if (b > 0)  { a = b; }
    else    { a = 2 * b; }
a = a + 1;
return a
```

```cpp
// flatten branching
int a = 0;
a = select(b > 0, b, 2 * b); // a = b > 0 ? b : 2 * b
a = a + 1;
return a;
```

</div>

2. Elimate mutable local variables: apply a series of local variable store forwarding transforms

<div grid="~ cols-2 gap-10">

```cpp
// flatten branching
int a = 0;
a = select(b > 0, b, 2 * b);
a = a + 1;
return a;
```

```cpp
// eliminate mutable var
ssa1 = select(b > 0, b, 2 * b);
ssa2 = ssa1 + 1;
return ssa2;
```

</div>

After IR transfroms, it only has to differentiate the straight-line code without mutable variables.

</p>

<!--

* kernel 内的 AD 主要由 preprocessing passes 和 make adjoint pass 组成
* preprocessing passes 的主要任务是将原始的 taichi ir 转换为直线型的 ssa ir，具体进行两个操作
    1. 分支扁平化：使用三元运算符 `select` 代替原始 ir 中的 `if` 控制流
    2. 消除可变的局部变量：创建新的变量来替代可变的变量，直至 ir 不再有可变变量也就是每个变量都成为 ssa
* 经过上述的两个 transform 后 diff taichi 就可以对简化后的直线型 ssa ir 进行反向的 AD

-->

---

# Local AD: Make Adjoint

<hr color="lightgray">

<p>

**Make adjoint pass** transform a forward (primal) kernel into its gradient (adjoint) kernel

<div class="my-2rem">

$$
\mathbf{(primal) \ } f(X) = Y \xrightarrow{\text{Reverse AD}} \mathbf{(adjoint) \ } f^{*}(X, Y^{*}) = X^{*}
$$

</div>

Consider $y_i = \sin x_i^2$ which has a primal kernel like

<img src="/images/primal.png" class="mx-auto w-sm" />

</p>

<!--

* make adjoint pass 的目的是根据原始函数生成伴随函数，其形式是接收原始输入和输出的伴随来计算输出的伴随
* 用一个具体的例子帮助理解，下面是 y = sin(x^2) 的原始 ir
    1. 取 x
    2. 计算 x^2
    3. 计算 sin(x^2)
    4. 将 sin(x^2) 存入 y

-->

---

# Local AD: Make Adjoint (cont'd)

<hr color="lightgray">

<p>

Each SSA instruction is allocated with a local adjoint variable for grad contribution accumulation.

<img src="/images/make-adjoint.svg" class="mx-auto h-sm" />

</p>

<!--

* 在 make adjoint pass 中正向的计算代码将被保留，同时为每个 ssa 分配变量用于存储其伴随 (1adj ~ 3adj)
* 正向代码的下方是对应的反向 AD 代码，使用链式法则将输出的伴随逐个 op 地传递到输入
    * 从全局 tensor 中取输出 y 的伴随
    * y = sin(x^2), %3 = sin(x^2), dy/d3 = 1，所以 3_adj 累加 1 * y_adj
    * %2 = x^2, d3/d2 = 2xcos(x^2)，所以 2_adj 累加 2xcos(x^2) * 3_adj
    * %1 = x, d2/d1 = 2x，所以 1_adj 累加 2x * 2_adj
    * 将输入 x 的伴随存入全局 tensor 中

-->

---

# Local AD: Loops & Parallelism

<hr color="lightgray">

<br>

<p>

### Loops

<div class="my-1rem">

* Parallel loops are preserved while other loops are reversed during AD transforms
* Loops with mutating local variables are not supported since that would require a complex and costly runtime stack

</div>

<br>

### Parralleism and Thread Safety

<div class="my-1rem">

Programmers should use atomic operations that can be automatic differentiated by the system.

</div>

</p>

<!--

* 在 precessing passes 中完成了分支展平和消除 mut，此外还有几个方面是需要加以考虑的
    * diff taichi 将循环分为了并行循环和非并行循环，在反向过程中并行循环将被保留，而非并行循环将会被逆转（见后一页例子）
    * diff taichi 中的循环都是完美嵌套循环，且不允许循环带有局部变量，因为这需要一个昂贵的 runtime stack 保持每个局部变量的状态
* 为了保证线程安全，用户必须使用可微分的原子 op 实现 kernel，这与之前对 AD 的限制也是一致的

-->

---

# Global AD: Light-weight Tape

<hr color="lightgray">

<p>

DiffTaichi has a light-weight tape only records kernel names and the input parameters.

<img src="/images/tape.png" class="mx-auto my-2rem" />

All intermediate results are stored in global tensors and DiffTaichi AD is evaluating gradients w.r.t. input global tensors instead of the input parameters.

</p>

<!--

* 对于 kernels 之间即 e2e 的 AD，diff taichi 使用一个轻量级的 tape 来保存中间变量的状态
* 与其他 AD 系统不同的是，diff taichi 的 tape 只保存 kernel name 和 kernel inputs，而所有的中间变量值都存储在全局 tensor 中
* diff taichi 生成梯度的依据是全局 tensor 中的值而不是 kernel 的输入
* 图中是一个非并行循环的例子，在反向过程中 tape 会自动逆转生成梯度的次序

-->

---

# Global AD: Learning and Optimization

<hr color="lightgray">

<p>

A [simple mass-spring example](https://github.com/taichi-dev/difftaichi/blob/master/examples/mass_spring_simple.py) uses automatic differentation for model optimization.

<img src="/images/mass-spring.png" class="mx-auto w-xl" />

`ti.Tape` will automatically record and reverse gradients for backpropagation.

<img src="/images/spring-rest-length-optimization.png" class="mx-auto w-xl" />

</p>

<!--

* 有 tape 这个数据结构就可以很方便的进行 e2e 的 AD，文中作者用 mass-spring 这个例子演示了如何使用 diff taichi 来进行模型的优化
    * 定义 `forward` 函数根据胡克定律来模拟质点的运动
    * 定义损失函数用最小二乘法来计算 loss
    * 组合 `forward` 和 `compute_loss` 并利用梯度下降来更新弹簧长度

-->

---

# Global AD: Complex Kernels

<hr color="lightgray">

<p>

DiffTaichi provides decorators `ti.complex_kernel` and `ti.complex_kernel_grad` for users who want to overwrite default automatic differentiation provided by the compiler.

```python
@ti.complex_kernel
def advance(s):
    clear_grid()
    compute_actuation(s)
    p2g(s)
    grid_op()
    g2p(s)

@ti.complex_kernel_grad(advance)
def advance_grad(s):
    clear_grid()
    p2g(s)
    grid_op() # recompute the grip

    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)
    compute_actuation.grad(s)
```
</p>

<!--

* 如果用户不满足于 diff taichi 提供的默认 AD，diff taichi 也提供了一套机制让用户自定义 kernel 的微分
    * 在比较老的版本中，用户可以使用 `ti.complex_kernel` 装饰器定义一个正向 complex kernel，用 `ti.complex_kernel_grad` 装饰器定义某个正向 complex kernel 的微分
    * 在新版本中 api 分别修改为 `ti.ad.grad_replaced` 和 `ti.ad.grad_for`
* 屏幕上是使用 complex kernel 机制实现 checkpoint 的例子，在反向过程中 `grid_op` 是一个 checkpoint，在这个位置将进行 grid 的重计算以便后续 grad op 的使用

-->

---

# Global AD: Checkpointing

<hr color="lightgray">

<p>

Checkpointing can be used with customized complex kernels to **reduce memory usage** by recomputation.

<img src="/images/checkpoint.svg" class="mx-auto w-md my-2rem" />

Simulation with $O(n)$ steps can be split into many segments of $S$ steps that store only the first simulation state in each segment.

Total space consumption is $O(S + n/S)$: $O(n/S)$ for checkpoints and $O(S)$ for recomputation, and the time complexity remains $O(n)$.

</p>

<!--

* 利用 complex kernel 和 checkpoint 可以在 diff taichi 中实现 segment-wise 的重计算
    * 对于有 O(n) 个步骤的物理模拟，如果朴素地保存每一个步骤的状态需要 O(n) 的空间复杂度
    * 使用 checkpoint 将每 S 个步骤分为一段，每一段都只保存第一个步骤，其他的步骤通过重计算获得
        * 这将存储 e2e 的空间复杂度降低到 O(n/S)
        * 在每一段内进行重计算时需要 O(S) 的空间复杂度，故总的空间复杂度为 O(n/S + S)，可以得出 S = O(√n) 时总时间复杂度最低，为 O(√n)
        * 重计算过程没有引入冗余计算，因此时间复杂度仍为 O(n)

-->

---

# Evaluation: Elastic Objects

<hr color="lightgray">

<p>

<img src="/images/diffmpm80.gif" class="mx-auto" />

<img src="/images/elastic.png" class="mx-auto mt-2rem" />

</p>

<!--

* 作者在 10 个不同的场景实现相同的物理模拟器来比较不同的工具 (diff taichi, tf, cuda) 之间的 performance 和 productivity 差距，文中选取了 10 个例子中的 3 个
* 第一个场景是弹性物体的模拟，作者以总用时来衡量三者之间的性能，用代码长度衡量三者之间的生产力
    * 三者中性能最好的是手写的 cuda kernel，比 diff taichi 快 8%，但代码长度是 diff taichi 的 4 倍还多
    * 为神经网络设计的 tf 在这一项对比中不仅性能最差而且开发效率也不及 diff taichi

-->

---

# Evaluation: Fluid Simulator

<hr color="lightgray">

<p>

<img src="/images/liquid.gif" class="mx-auto w-lg" />

<img src="/images/fluid.png" class="mx-auto mt-2rem w-2xl" />

</p>

<!--

* 第二个场景是流体的模拟，这次 diff taichi 与另外三个流行的 AD 框架 pytorch, jax 和 autograd 进行比较
    * 代码长度最短的是 autograd，但它的用时是最长的，可能的原因是 autograd 使用 64 位浮点数使得运行时间翻倍
    * GPU 上的 diff taichi 和 jax 都取得不错的性能，和 jax 相比，diff taichi 在 jit 上花费 2s，而 jax 的 jit 编译需要 2min

-->

---

# Evaluation: Rigid Body Simulator

<hr color="lightgray">

<p>

<img src="/images/rb_final2.gif" class="mx-auto h-10rem" />

Problem: naively differentiating leads to misleading gradients due to rigid body collisions

<img src="/images/rigid.png" class="mx-auto mt-rem w-md" />

</p>

<!--

* 第三个场景稍有不同，在对刚体的模拟中作者比较的是朴素的模拟方法和引入碰撞时间 TOI 的模拟方法
    * 在朴素方法中存在一个问题：时间离散化会导致在某个点可能产生错误的梯度

-->

---

# Evaluation: Rigid Body Simulator (cont'd)

<hr color="lightgray">

<p>

Solution: adding continuous collision resolution which considers precise time of impact (TOI)

<img src="/images/toi.png" class="mx-auto w-2xl" />

TOI barely improves the forward simulation but the gradient will be corrected effectively

<img src="/images/rigid-comparison.png" class="mx-auto w-2xl" />

</p>

<!--

* 作者的解决方案是考虑碰撞时间 TOI，这种方式对正向模拟没有影响，但有效地校正了反向的梯度
* 上图是对比 5 次实验中朴素模拟和引入 TOI 的模拟对 loss 的影响，在朴素模拟中 loss 一直在较高水平范围内震荡，而引入 TOI 的模拟 loss 会在几次迭代内很快地下降
* 下图是朴素方法和引入 TOI 方法的模拟及 loss 和 gradient 的比较
    * 在朴素方法中由于错误的梯度的存在，导致蓝色球在起始位置和最终位置都低于绿色球，呈现出反直觉的现象，引入 TOI 后起始位置更低的蓝色球最终位置高于绿色球
    * 第三幅图展示两种方法计算的梯度，朴素方法得到错误的梯度 1，引入 TOI 后得到正确的梯度 -1
    * 第四幅图是第三幅图中间部分的 zoom in，朴素方法的 loss 一直在某个范围内震荡，而引入 TOI 的方法 loss 可以正确下降

-->

---
layout: center
---

# <p class="text-6xl">THANK YOU</p>