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
from cutlass.utils.profiler import GpuTimer

from cutlass.backend import compiler
from cutlass.backend.library import TensorDescription, TileDescription


class TensorInfo:
    def __init__(self, name, shape, dtype, layout):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.layout = layout

    def __repr__(self):
        return f"TensorInfo(name={self.name}, shape={self.shape}, dtype={self.dtype}, layout={self.layout})"


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
        kernel_name: str,
        input_tensor: List[TensorInfo],
        output_tensor: List[TensorInfo],
    ):
        self.gemm_plan = gemm_plan
        self.kernel_name = kernel_name
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self._init_gemm_default_tensors()

    def _init_gemm_default_tensors(self):
        for out_tensor in self.output_tensor:
            if out_tensor.name == "D":
                self.D = out_tensor
                self.M = out_tensor.shape[-2]
                self.N = out_tensor.shape[-1]
                if len(out_tensor.shape) == 3:
                    self.L = out_tensor.shape[0]
                else:
                    self.L = 1

        for input_tensor in self.input_tensor:
            if input_tensor.name == "A":
                self.A = input_tensor
                if self.A.layout == cutlass.LayoutType.RowMajor:
                    self.K = input_tensor.shape[-1]
                else:
                    self.K = input_tensor.shape[-2]
                self.shape_A = input_tensor.shape

            if input_tensor.name == "B":
                self.B = input_tensor
                if self.B.layout == cutlass.LayoutType.RowMajor:
                    K = input_tensor.shape[-2]
                else:
                    K = input_tensor.shape[-1]
                assert K == self.K
                self.shape_B = input_tensor.shape

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
        # or ["ptr_row", "null_default", "dRow"]
        # or ["ptr_col", "null_default", "dCol"]

        return (
            field_names == ["ptr_aux", "null_default", "dAux"]
            or field_names == ["ptr_row", "null_default", "dRow"]
            or field_names == ["ptr_col", "null_default", "dCol"]
        )

    def input_tensor_broadcast_type(self, cls):
        if not hasattr(cls, "_fields_"):
            return False
        fields = cls._fields_
        field_names = [field[0] for field in fields]
        return field_names[2][1:]

    def is_output_tensor_structure(self, cls):
        if not hasattr(cls, "_fields_"):
            return False
        fields = cls._fields_
        field_names = [field[0] for field in fields]
        # field_names should be ["ptr_aux", "dAux"]
        return field_names == ["ptr_aux", "dAux"]

    def is_reduce_structure(self, cls):
        if not hasattr(cls, "_fields_"):
            return False
        fields = cls._fields_
        field_names = [field[0] for field in fields]
        return field_names == ["ptr", "reduce_identity", "dMNL"]

    def emit_tensor_structure_str(self, cls, field_name):
        o = cls({field_name: None})
        res = [
            f"(Element{field_name} *)ptr_{field_name}",
            str(o.reduce_identity),
            self.emit_tuple_type(o.dMNL),
        ]
        return "{" + ", ".join(res) + "}"
    
    def emit_tuple_type(self, tuple):
        if type(tuple) == cutlass.backend.c_types.EmptyByte:
            return "{}"
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

    def emit_input_tensor_structure_str(self, cls, field_name, broadcast_type="Aux"):
        o = cls({field_name: None})
        tuple_value = getattr(o, "d" + broadcast_type)
        res = [
            f"(Element{field_name} *)ptr_{field_name}",
            f"(Element{field_name} ){o.null_default}",
            self.emit_tuple_type(tuple_value),
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
                broadcast_type = self.input_tensor_broadcast_type(cls)
                return indent + self.emit_input_tensor_structure_str(
                    cls, field_name, broadcast_type
                )
            elif self.is_output_tensor_structure(cls):
                return indent + self.emit_output_tensor_structure_str(cls, field_name)
            elif self.is_reduce_structure(cls):
                return indent + self.emit_tensor_structure_str(cls, field_name)

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
        epilogue_type = self.gemm_plan.epilogue_visitor.epilogue_type
        assert len(epilogue_type._fields_) == 1
        assert epilogue_type._fields_[0][0] == "output_op"
        # print(self.ctypes_to_cpp_constructor(epilogue_type._fields_[0][1]))
        return self.ctypes_to_cpp_constructor(epilogue_type._fields_[0][1])

    def evt_problem_size(self):
        template = """
static int M = ${M};
static int N = ${N};
static int K = ${K};
static int L = ${L};

cutlass::gemm::GemmCoord problem_size(M, N, K);
        """

        return SubstituteTemplate(
            template, {"M": self.M, "N": self.N, "K": self.K, "L": self.L}
        )

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

        batch_stride_A = self.shape_A[-1] * self.shape_A[-2]
        batch_stride_B = self.shape_B[-1] * self.shape_B[-2]

        stride_A = self.shape_A[-1]
        stride_B = self.shape_B[-1]

        gemm_arguments = SubstituteTemplate(
            gemm_arguments_template,
            {
                "L": self.L,
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
                "kernel_name": self.kernel_name,
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
        # temp_cu = tempfile.NamedTemporaryFile(
        #     prefix="kernel", suffix=".cu", delete=False
        # )
        # delete .so
        cu_name = self.shared_lib_name[:-3] + ".cu"

        with open(cu_name, "w") as f:
            f.write(self.cutlass_code)

        cmd = SubstituteTemplate(
            self.cmd_template,
            {"shared_lib_name": self.shared_lib_name, "cu_name": cu_name},
        )
        # print(cmd)
        cmd = cmd.split(" ")

        error_file = "error.log"
        self.compile_with_nvcc(cmd, cu_name, error_file)


def emit_cutlass_evt_kernel(
    gemm_plan: cutlass.op.Gemm,
    kernel_name: str,
    input_tensor: List[TensorInfo],
    output_tensor: List[TensorInfo],
    shared_lib_name: str,
):
    emitter = CutlassEvtEmitter(gemm_plan, kernel_name, input_tensor, output_tensor)
    cutlass_code = emitter.emit()
    compiler = Compiler(cutlass_code, shared_lib_name)
    compiler.compile()


def dtype_to_torch(dtype):
    if dtype == "f16":
        return torch.float16
    elif dtype == "f32":
        return torch.float32
    elif dtype == "bf16":
        return torch.bfloat16
    else:
        raise ValueError("Unsupported data type")


def run_kernel(input_tensors, output_tensors, kernel_name, so_name):
    cutlass_lib = ctypes.CDLL("./" + so_name)
    # find func kernel_name
    cutlass_kernel = getattr(cutlass_lib, kernel_name)
    cutlass_kernel.argtypes = [ctypes.c_void_p] * len(
        input_tensors + output_tensors
    ) + [
        ctypes.c_void_p
    ]  # stream
    cutlass_kernel.restype = None

    all_torch_tensors = []
    # prepare data
    for tensor in input_tensors + output_tensors:
        all_torch_tensors.append(
            torch.randn(tensor.shape, dtype=tensor.dtype, device="cuda").uniform_(-1, 1)
        )

    # call kernel
    cutlass_kernel(
        *[tensor.data_ptr() for tensor in all_torch_tensors] + [ctypes.c_void_p(0)]
    )


def profile_kernel(input_tensors, output_tensors, kernel_name, so_name, num_iter=10):
    cutlass_lib = ctypes.CDLL("./" + so_name)
    # find func kernel_name
    cutlass_kernel = getattr(cutlass_lib, kernel_name)
    cutlass_kernel.argtypes = [ctypes.c_void_p] * len(
        input_tensors + output_tensors
    ) + [
        ctypes.c_void_p
    ]  # stream
    cutlass_kernel.restype = None

    all_torch_tensors = []
    # prepare data
    for tensor in input_tensors + output_tensors:
        all_torch_tensors.append(
            torch.randn(tensor.shape, dtype=tensor.dtype, device="cuda").uniform_(-1, 1)
        )

    # warm up
    for _ in range(10):
        cutlass_kernel(
            *[tensor.data_ptr() for tensor in all_torch_tensors] + [ctypes.c_void_p(0)]
        )
    torch.cuda.synchronize()
    # run and return the average time
    timer = GpuTimer()
    timer.start()
    for _ in range(num_iter):
        cutlass_kernel(
            *[tensor.data_ptr() for tensor in all_torch_tensors] + [ctypes.c_void_p(0)]
        )
    timer.stop_and_wait()
    return timer.duration(num_iter)
