from emitter import emit_cutlass_evt_kernel
from emitter import TensorInfo

import torch
import argparse
import cutlass
from cutlass import Tensor as FakeTensor

kernel_name = "gemm_add_bias"
so_name = kernel_name + ".so"


def dtype_to_torch(dtype):
    if dtype == "f16":
        return torch.float16
    elif dtype == "f32":
        return torch.float32
    elif dtype == "bf16":
        return torch.bfloat16
    else:
        raise ValueError("Unsupported data type")


if __name__ == "__main__":
    # parse M, N, K, and data type

    parser = argparse.ArgumentParser(
        description="Generate CUTLASS EVT kernel for GEMM with bias addition"
    )
    parser.add_argument("--L", type=int, default=1, help="Number of GEMM operations")
    parser.add_argument("--M", type=int, default=128, help="M dimension of GEMM")
    parser.add_argument("--N", type=int, default=128, help="N dimension of GEMM")
    parser.add_argument("--K", type=int, default=128, help="K dimension of GEMM")
    parser.add_argument(
        "--input_dtype", type=str, default="f16", help="Data type of GEMM"
    )
    parser.add_argument(
        "--accum_dtype", type=str, default="f32", help="Data type of accumulation"
    )

    args = parser.parse_args()
    L = args.L
    M = args.M
    N = args.N
    K = args.K
    input_dtype = args.input_dtype
    accum_dtype = args.accum_dtype

    type_accum = dtype_to_torch(accum_dtype)
    type_input = dtype_to_torch(input_dtype)
    layout_type = cutlass.LayoutType.RowMajor
    A_TensorInfo = TensorInfo("A", (L, M, K), type_input, layout_type)
    B_TensorInfo = TensorInfo("B", (L, K, N), type_input, layout_type)
    C_TensorInfo = TensorInfo("C", (L, M, N), type_input, layout_type)
    D_TensorInfo = TensorInfo("D", (L, M, N), type_input, layout_type)

    input_tensors = [A_TensorInfo, B_TensorInfo, C_TensorInfo]
    output_tensors = [D_TensorInfo]

    def example_epilogue(accum, C):
        D = accum * 0.2 + C * 0.8
        return D

    example_tensors = {
        "accum": FakeTensor(
            element=type_accum, shape=(L, M, N), layout_tag=layout_type
        ),
    }

    for tensor in input_tensors + output_tensors:
        example_tensors[tensor.name] = FakeTensor(
            element=tensor.dtype, shape=tensor.shape, layout_tag=layout_type
        )

    plan = cutlass.op.Gemm(
        element=D_TensorInfo.dtype,
        layout=D_TensorInfo.layout,
        element_accumulator=type_accum,
    )

    epilogue_visitor = cutlass.epilogue.trace(example_epilogue, example_tensors)
    plan.epilogue_visitor = epilogue_visitor

    emit_cutlass_evt_kernel(
        plan, kernel_name, input_tensors, output_tensors, so_name
    )


    
