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
* Hand-writing CUDA kernels takes long time

<img src="/images/productivity-performance.png" class="mx-auto w-lg" />

</p>

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

---

# Assumption

<hr color="lightgray">

<p>

The goal of AD is clear:

<div class="my-2rem">

$$
f(X) = Y \xrightarrow{\text{AD}} f^{*}(X, Y^{*}) = X^{*}
$$

</div>

Some assumptions on imperative kernels are made to avoid overwriting each other's output:

1. If a global tensor element is written more than once, then starting from the second write, the write must come **in the form of an atomic add ("accumulation")**
2. No read access happen to a global tensor element **until its accumulation is done**

</p>

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

<img src="/images/primal.png" class="mx-auto" />

</p>

---

# Local AD: Make Adjoint (cont'd)

<hr color="lightgray">

<p>

Each SSA instruction is allocated with a local adjoint variable for grad contribution accumulation.

<img src="/images/adjoint.png" class="mx-auto h-sm" />

</p>

---

# Local AD: Loops & Parallelism

<hr color="lightgray">

<p>

### Loops

<div class="my-1rem">

* Parallel loops are preserved while other loops are reversed during AD transforms
* Loops with mutating local variables are not supported since that would require a complex and costly runtime stack

</div>

### Parralleism and Thread Safety

<div class="my-1rem">

Programmers should use atomic operations that can be automatic differentiated by the system.

</div>

</p>

---

# Global AD: Light-weight Tape

<hr color="lightgray">

<p>

DiffTaichi has a light-weight tape only records kernel names and the input parameters.

<img src="/images/tape.png" class="mx-auto my-2rem" />

All intermediate results are stored in global tensors and DiffTaichi AD is evaluating gradients w.r.t. input global tensors instead of the input parameters.

</p>

---

# Global AD: Learning and Optimization

<hr color="lightgray">

<p>

A [simple mass-spring example](https://github.com/taichi-dev/difftaichi/blob/master/examples/mass_spring_simple.py) uses automatic differentation for model optimization.

<img src="/images/mass-spring.png" class="mx-auto w-xl" />

`ti.Tape` will automatically record and reverse gradients for backpropagation.

<img src="/images/spring-rest-length-optimization.png" class="mx-auto w-xl" />

</p>

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

---

# Global AD: Checkpointing

<hr color="lightgray">

<p>

Checkpointing can be used with customized complex kernels to **reduce memory usage** by recomputation.

<img src="/images/checkpoint.svg" class="mx-auto w-md my-2rem" />

Simulation with $O(n)$ steps can be split into many segments of $S$ steps that store only the first simulation state in each segment.

Total space consumption is $O(S + n/S)$: $O(n/S)$ for checkpoints and $O(S)$ for recomputation, and the time complexity remains $O(n)$.


</p>

---

# Evaluation: Elastic Objects

<hr color="lightgray">

<p>

<img src="/images/diffmpm80.gif" class="mx-auto" />

<img src="/images/elastic.png" class="mx-auto mt-2rem" />

</p>

---

# Evaluation: Fluid Simulator

<hr color="lightgray">

<p>

<img src="/images/liquid.gif" class="mx-auto w-lg" />

<img src="/images/fluid.png" class="mx-auto mt-2rem w-2xl" />

</p>

---

# Evaluation: Rigid Body Simulator

<hr color="lightgray">

<p>

<img src="/images/rb_final2.gif" class="mx-auto h-15rem" />

<img src="/images/rigid.png" class="mx-auto mt-2rem w-2xl" />

</p>

---
layout: center
---

# <p class="text-6xl">THANK YOU</p>