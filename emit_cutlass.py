from typing import List
import torch
import os
import ctypes
import re
import subprocess
import tempfile


import cutlass
from cutlass.epilogue import relu
from cutlass import Tensor as FakeTensor
from cutlass.utils.profiler import CUDAEventProfiler

from cutlass.backend import compiler
from cutlass.backend.library import TensorDescription, TileDescription

kernel_name = "cutlass_kernel"

# GEMM Problem Size
L = 1
M = 1024
N = 1024
K = 1024

# Data Types
type_A = torch.float16
type_B = torch.float16
type_C = torch.float16
type_D = torch.float16

type_accumulator = torch.float32

layout_A = cutlass.LayoutType.RowMajor
layout_B = cutlass.LayoutType.RowMajor
layout_C = cutlass.LayoutType.RowMajor
layout_D = cutlass.LayoutType.RowMajor

shape_A = (L, M, K)
shape_B = (L, K, N)
shape_C = (L, M, N)
shape_D = (L, M, N)


class TensorInfo:
    def __init__(self, name, shape, dtype, layout):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.layout = layout

    def __repr__(self):
        return f"TensorInfo(name={self.name}, shape={self.shape}, dtype={self.dtype}, layout={self.layout})"


input_tensor = [
    TensorInfo("A", shape_A, type_A, layout_A),
    TensorInfo("B", shape_B, type_B, layout_B),
    TensorInfo("C", shape_C, type_C, layout_C),
]

output_tensor = [
    TensorInfo("D", shape_D, type_D, layout_D),
]

plan = cutlass.op.Gemm(
    element=type_D, layout=layout_D, element_accumulator=type_accumulator
)


# Define epilogue visitor
def example_epilogue(accum, C):
    D = accum * 0.2 + C * 0.8
    return D


# define example tensors
example_tensors = {
    "accum": FakeTensor(
        element=torch.float32, shape=(M, N), layout_tag=cutlass.LayoutType.RowMajor
    ),
    "C": FakeTensor(
        element=type_C, shape=(M, N), layout_tag=cutlass.LayoutType.RowMajor
    ),
    "D": FakeTensor(
        element=type_D, shape=(M, N), layout_tag=cutlass.LayoutType.RowMajor
    ),
}

# Trace the epilogue visitor
epilogue_visitor = cutlass.epilogue.trace(example_epilogue, example_tensors)
plan.epilogue_visitor = epilogue_visitor


def SubstituteTemplate(template, values):
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            value = str(value)
            regex = "\\$\\{%s\\}" % key
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text


def torch_dtype_to_cutlass(dtype):
    if dtype == torch.float16:
        return "cutlass::half_t"
    elif dtype == torch.bfloat16:
        return "cutlass::bfloat16_t"
    elif dtype == torch.float32:
        return "float"
    elif dtype == torch.int8:
        return "int8_t"
    elif dtype == torch.int32:
        return "int32_t"
    elif dtype == torch.int64:
        return "int64_t"
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def cutlass_layout_to_str(layout):
    if layout == cutlass.LayoutType.RowMajor:
        return "cutlass::layout::RowMajor"
    elif layout == cutlass.LayoutType.ColumnMajor:
        return "cutlass::layout::ColumnMajor"
    else:
        raise ValueError(f"Unsupported layout {layout}")


def get_alignment(dim):
    if dim % 8 == 0:
        return 8
    elif dim % 4 == 0:
        return 4
    elif dim % 2 == 0:
        return 2
    else:
        return 1


class CutlassEvtEmitter:
    def __init__(
        self,
        gemm_plan: cutlass.op.Gemm,
        input_tensor: List[TensorInfo],
        output_tensor: List[TensorInfo],
    ):
        self.gemm_plan = gemm_plan
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def is_imm_structure(self, cls):
        if not hasattr(cls, "_fields_"):
            return False
        fields = cls._fields_
        field_names = [field[0] for field in fields]
        # field_names should be ["scalars, "scalar_ptrs", "dScalar"]
        return field_names == ["scalars", "scalar_ptrs", "dScalar"]

    def is_input_tensor_structure(self, cls):
        if not hasattr(cls, "_fields_"):
            return False
        fields = cls._fields_
        field_names = [field[0] for field in fields]
        # field_names should be ["ptr_aux", "null_default", "dAux"]
        return field_names == ["ptr_aux", "null_default", "dAux"]

    def is_output_tensor_structure(self, cls):
        if not hasattr(cls, "_fields_"):
            return False
        fields = cls._fields_
        field_names = [field[0] for field in fields]
        # field_names should be ["ptr_aux", "dAux"]
        return field_names == ["ptr_aux", "dAux"]

    def emit_tuple_type(self, tuple):
        fields = tuple._fields_
        res = []
        for field in fields:
            field_name, field_type = field
            if field_type == cutlass.backend.c_types.Empty:
                res.append("{}")
            else:
                stride = getattr(tuple, field_name)
                res.append(str(stride))
        return "{" + ", ".join(res) + "}"

    def emit_input_tensor_structure_str(self, cls, field_name):
        o = cls({field_name: None})
        res = [
            f"(Element{field_name} *)ptr_{field_name}",
            f"(Element{field_name} ){o.null_default}",
            self.emit_tuple_type(o.dAux),
        ]
        return "{" + ", ".join(res) + "}"

    def emit_output_tensor_structure_str(self, cls, field_name):
        o = cls({field_name: None})
        res = [f"(Element{field_name} *)ptr_{field_name}", self.emit_tuple_type(o.dAux)]
        return "{" + ", ".join(res) + "}"

    def emit_imm_structure_str(self, cls):
        o = cls({})
        scarlars = str(o.scalars)
        scalar_ptrs = "{}" if o.scalar_ptrs == None else str(o.scalar_ptrs)
        dScalar = "{}"
        return f"{{ {scarlars}, {scalar_ptrs}, {dScalar} }}"

    def ctypes_to_cpp_constructor(self, cls):
        def translate_field(field_name, field_type, level):
            # Define how different ctypes should be represented in the output
            if issubclass(field_type, ctypes.Structure):
                return generate_cpp_constructor(
                    field_type, field_name, level
                )  # Recursively handle nested structures
            elif field_type == ctypes.c_byte:
                return "{}"
            elif field_type in (
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_double,
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.c_long,
                ctypes.c_uint16,
            ):
                return field_name  # Use the field's name
            else:
                return "unknown_field"

        def generate_cpp_constructor(cls, field_name="", level=0):
            lines = []
            indent = "  " * level
            if not cls._fields_:
                return f"{indent}{{}}"
            elif cls == cutlass.backend.c_types.EmptyByte:
                return indent + "{}"
            elif self.is_imm_structure(cls):
                return indent + self.emit_imm_structure_str(cls)
            elif self.is_input_tensor_structure(cls):
                return indent + self.emit_input_tensor_structure_str(cls, field_name)
            elif self.is_output_tensor_structure(cls):
                return indent + self.emit_output_tensor_structure_str(cls, field_name)

            lines.append(f"{indent}{{")
            for field_name, field_type in cls._fields_:
                field_value = translate_field(field_name, field_type, level + 1)
                if issubclass(field_type, ctypes.Structure):
                    lines.append(f"{field_value},  // {field_name}")
                else:
                    lines.append(f"{indent}  {field_value},  // {field_name}")
            lines.append(f"{indent}}}")

            return "\n".join(lines)

        return generate_cpp_constructor(cls)

    def emit_incldue(self):
        include_list = [
            "builtin_types.h",
            "device_launch_parameters.h",
            "stddef.h",
            "cutlass/cutlass.h",
            "cutlass/gemm_coord.h",
            "cutlass/numeric_types.h",
            "cutlass/arch/arch.h",
            "cutlass/arch/mma.h",
            "cutlass/layout/matrix.h",
            "cutlass/gemm/device/gemm.h",
            "cutlass/gemm/kernel/default_gemm_universal.h",
            "cutlass/epilogue/threadblock/fusion/visitors.hpp",
            "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h",
            "cutlass/gemm/device/gemm_universal_adapter.h",
            "cutlass/util/device_memory.h",
        ]
        return "\n".join([f'#include "{include}"' for include in include_list])

    def emit_evt_argument_type(self):
        epilogue_type = plan.epilogue_visitor.epilogue_type
        assert len(epilogue_type._fields_) == 1
        assert epilogue_type._fields_[0][0] == "output_op"
        # print(self.ctypes_to_cpp_constructor(epilogue_type._fields_[0][1]))
        return self.ctypes_to_cpp_constructor(epilogue_type._fields_[0][1])

    def evt_problem_size(self):
        template = """
int M = ${M};
int N = ${N};
int K = ${K};
int L = ${L};

cutlass::gemm::GemmCoord problem_size(M, N, K);
        """

        return SubstituteTemplate(template, {"M": M, "N": N, "K": K, "L": L})

    def emit_type_and_layout(self):
        template_type = "using Element${tensor_name} = ${tensor_type};\n"
        template_layout = "using Layout${tensor_name} = ${tensor_layout};\n"

        type_str = ""
        layout_str = ""

        for tensor in self.input_tensor + self.output_tensor:
            type_str += SubstituteTemplate(
                template_type,
                {
                    "tensor_name": tensor.name,
                    "tensor_type": torch_dtype_to_cutlass(tensor.dtype),
                },
            )
            layout_str += SubstituteTemplate(
                template_layout,
                {
                    "tensor_name": tensor.name,
                    "tensor_layout": cutlass_layout_to_str(tensor.layout),
                },
            )

        return type_str + layout_str

    def emit_cutlass_kernel_declaration(
        self,
        tile_description: TileDescription = None,
        alignment_A: int = None,
        alignment_B: int = None,
        alignment_C: int = None,
    ):

        self.gemm_plan.operation = self.gemm_plan.construct(
            tile_description, alignment_A, alignment_B, alignment_C
        )
        rt_module_source = self.gemm_plan.operation.rt_module.emit()
        kernel_name = self.gemm_plan.operation.rt_module.name()
        device_gemm = f"using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<{kernel_name}_type>;"
        return "\n".join([rt_module_source, device_gemm])

    def emit_cutlass_entry_function(self):
        gemm_arguments_template = """
  typename GemmOp::Arguments argument(
      cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
      problem_size,                                // problem_size
      ${L},                                        // batch count
      callback_args,                            // epilogue parameters
      ptr_A,                                    // ptr_A
      ptr_B,                                    // ptr_B
      nullptr,                                  // ptr_C (unused)
      nullptr,                                  // ptr_D (unused)
      ${batch_stride_A},                        // batch_stride_A
      ${batch_stride_B},                        // batch_stride_B
      0,                                        // batch_stride_C
      0,                                        // batch_stride_D
      ${stride_A},                              // stride_a
      ${stride_B},                              // stride_b
      0,                                        // stride_c
      0,                                        // stride_d
      nullptr, nullptr, nullptr                 // gather...
  );
        """

        func_params = [
            "void *" + "ptr_" + tensor.name
            for tensor in self.input_tensor + self.output_tensor
        ]

        func_params += [
            "cudaStream_t stream",
        ]

        EVT_D_callbacks = (
            "typename EVTD::Arguments callback_args"
            + self.emit_evt_argument_type()
            + ";"
        )

        batch_stride_A = shape_A[-1] * shape_A[-2]
        batch_stride_B = shape_B[-1] * shape_B[-2]

        stride_A = shape_A[-1]
        stride_B = shape_B[-1]

        gemm_arguments = SubstituteTemplate(
            gemm_arguments_template,
            {
                "L": L,
                "batch_stride_A": batch_stride_A,
                "batch_stride_B": batch_stride_B,
                "stride_A": stride_A,
                "stride_B": stride_B,
            },
        )
        op_def_and_run = """
    // Define Gemm operator
    GemmOp gemm_op;

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = GemmOp::get_workspace_size(argument);
    if (workspace_size)
    {
        std::cout << "Workspace size: " << workspace_size << std::endl;
        return;
    }

    // Alloc workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

    // Check the problem size is supported or not
    CUTLASS_CHECK(GemmOp::can_implement(argument));

    // Initialize CUTLASS Gemm kernel with the arguments and workspace pointer
    CUTLASS_CHECK(gemm_op.initialize(argument, workspace.get(), stream));

    // run gemm
    CUTLASS_CHECK(gemm_op(argument, workspace.get(), stream));
}
"""

        func_template = """
extern "C" void ${kernel_name}(${func_params}) {
    ${EVT_D_callbacks}
    ${arguemnts}
    ${op_def_and_run}
"""

        func = SubstituteTemplate(
            func_template,
            {
                "kernel_name": kernel_name,
                "func_params": ", ".join(func_params),
                "EVT_D_callbacks": EVT_D_callbacks,
                "arguemnts": gemm_arguments,
                "op_def_and_run": op_def_and_run,
            },
        )

        return func

    def emit(self):
        code = ""
        code += self.emit_incldue()
        code += self.emit_cutlass_kernel_declaration()
        code += self.evt_problem_size()
        code += self.emit_type_and_layout()
        code += self.emit_cutlass_entry_function()
        return code


# print(cutlass_evt_emitter.emit())
# cutlass_evt_emitter._get_evt_argument_type()


class Compiler:
    def __init__(self, cutlass_code, shared_lib_name):
        self.cutlass_code = cutlass_code
        self.shared_lib_name = shared_lib_name
        self.cmd_template = "nvcc -x cu -Xcompiler=-fpermissive -Xcompiler=-w -Xcompiler=-fPIC -std=c++17 --expt-relaxed-constexpr -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --include-path=/usr/local/cuda/include --include-path=/workspace/cutlass/python/cutlass_library/../../include --include-path=/workspace/cutlass/python/cutlass_library/../../tools/util/include --include-path=/workspace/cutlass/python/cutlass_library/../../python/cutlass/cpp/include -arch=sm_80 -shared -o ${shared_lib_name} ${cu_name} -lcudart -lcuda"

        # print(self.cmd_template)

    def compile_with_nvcc(self, cmd, source, error_file):
        succeed = True
        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            error_message = e.output.decode()
            with open(error_file, "w") as error_out:
                error_log = "Compilation error for the following kernel: \n"
                error_log += source
                error_log += "\nError Message:\n"
                error_log += error_message
                error_out.write(error_log)
            succeed = False
        if not succeed:
            # Print the error log to stdout if log level is set to warning or higher
            # verbosity. Otherwise, simply point to the error log file.
            raise Exception(f"Invalid Kernel. See '{error_file}' for details.")

    def compile(self):
        # use Temporary directory to store the generated cu file
        tempfile.tempdir = "./"
        temp_cu = tempfile.NamedTemporaryFile(
            prefix="kernel", suffix=".cu", delete=False
        )

        with open(temp_cu.name, "w") as f:
            f.write(self.cutlass_code)

        cmd = SubstituteTemplate(
            self.cmd_template,
            {"shared_lib_name": self.shared_lib_name, "cu_name": temp_cu.name},
        )
        # print(cmd)
        cmd = cmd.split(" ")

        error_file = "error.log"
        self.compile_with_nvcc(cmd, temp_cu.name, error_file)


cutlass_evt_emitter = CutlassEvtEmitter(plan, input_tensor, output_tensor)
code = cutlass_evt_emitter.emit()
compiler = Compiler(code, "cutlass_evt.so")
compiler.compile()
