---
author: ConvolutedDog
pubDatetime: 2025-05-09T15:22:00Z
modDatetime: 2025-05-09T15:22:00Z
title: Learning Roadmap for TVM
featured: true
draft: false
slug: learning-roadmap-for-tvm
tags:
  - TVM
description:
  Suggested roadmap for new users who want to deeply learn TVM.
---

<figure>
  <img
    src="https://mlc.ai/_images/dev-deploy-form.png"
    alt="Free Classic wooden desk with writing materials, vintage clock, and a leather bag. Stock Photo"
  />
</figure>

## Table of contents

## Getting Started with TVM Frontend

The TVM frontend is the first step in the **Machine Learning Compilation (MLC)** pipeline. It converts models from deep learning frameworks (e.g., PyTorch, TensorFlow, ONNX) into TVM’s intermediate representation (IR), enabling subsequent optimization and deployment.

### The MLC course and the course materials

I recommend the [MLC course](https://mlc.ai/summer22/) as a starting point for learning the TVM frontend. The course materials are available at [mlc.ai](https://mlc.ai/). I think this course is ideal because it offers:
 - Beginner-friendly explanations with a focus on practical workflow
 - Multi-framework support: Clear examples for PyTorch, TensorFlow, and more
 - Hands-on Jupyter Notebooks for interactive learning
 - Official & Up-to-date: Maintained by the TVM team

This course requires a minimum set of prerequisites in machine learning:
 - Python, familiarity with numpy.
 - Some background in one deep learning framework (e.g. PyTorch, TensorFlow, JAX)
 - Experiences in system programming (e.g. C/CUDA) would be beneficial but not required.

### What should we primarily focus on in this course?

I think we should learn two fundamental concepts from this course:
 - Tensor Program Abstraction (TensorIR)
 - Automatic Program Optimization

#### 1. Tensor Program Abstraction (TensorIR)
 - TensorIR is an universal intermediate representation for tensor computations in TVM.
 - Deep integration with C++ backend:
   - Directly maps to hardware primitives (e.g., GPU warp-level operations)
   - Enables zero-overhead interoperability with handwritten kernels
 - Example of TensorIR's low-level control:
    ```python
    import tvm
    from tvm.script import tir as T

    @tvm.script.ir_module
    class MM:
        @T.prim_func
        def main(
            A: T.Buffer(shape=(MATRIX_SIZE, MATRIX_SIZE), dtype=dtype),
            B: T.Buffer(shape=(MATRIX_SIZE, MATRIX_SIZE), dtype=dtype),
            C: T.Buffer(shape=(MATRIX_SIZE, MATRIX_SIZE), dtype=dtype),
        ):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            for i, j, k in T.grid(MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap(
                        kinds="SSR", bindings=[i, j, k], dtype="int64"
                    )
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    ```

#### 2. Automatic Program Optimization

TVM uses ML-driven optimization pipeline:
 - Uses statistical cost models to search thousands of possible kernel variants.
 - Applies transformations (e.g., loop tiling, vectorization) without manual intervention.
 - Example of auto-tuning:
    ```python
    from tvm import meta_schedule as ms
    
    mod = MM
    database = ms.tune_tir(
        mod=mod,
        target="llvm --num-cores=1",
        max_trials_global=64,
        num_trials_per_iter=64,
        work_dir="./tune_tmp",
    )
    ```

## Learning Tips

