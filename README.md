
# Cutlass Emitter Tool
## Overview
The Cutlass Emitter Tool is a Python-based utility designed to facilitate the generation and compilation of custom CUDA kernels using NVIDIA's CUTLASS library. This tool simplifies the process of defining, emitting, and executing highly optimized matrix multiplication (GEMM) operations by automating the creation of CUDA code, compiling it with NVCC, and running it efficiently on NVIDIA GPUs.
## Features
Emit device-level Cutlass code and help you manage the construction of GEMM arguments.
## How to Use
Define Tensor Information: Create instances of TensorInfo for each input and output tensor, specifying their name, shape, data type, and layout.

Initialize Emitter: Instantiate the CutlassEvtEmitter with a GEMM plan, kernel name, and tensor information.

Generate and Compile Code: Use the emit_cutlass_evt_kernel function to generate CUDA code and compile it into a shared library.

Run and Profile Kernel: Use the run_kernel function to execute the kernel, and the profile_kernel function to measure its performance.
## Examples
See `gemm_add_bias_gelu.py`.
