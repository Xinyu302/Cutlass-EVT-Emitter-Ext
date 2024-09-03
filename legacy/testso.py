import ctypes
import torch

# 加载.so文件
cutlass_lib = ctypes.CDLL('./cutlass_evt.so')

# 定义函数签名
cutlass_kernel = cutlass_lib.cutlass_kernel
cutlass_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
cutlass_kernel.restype = None



m = n = k = 1024

# A = torch.randn(m, k, dtype=torch.float16, device='cuda')
# B = torch.randn(k, n, dtype=torch.float16, device='cuda')
# C = torch.randn(m, n, dtype=torch.float16, device='cuda')
# D = torch.zeros_like(C)
# use uniform instead of randn
A = torch.ceil(torch.empty((m, k), dtype=torch.float16, device="cuda").uniform_(-1, 1))
B = torch.ceil(torch.empty((k, n), dtype=torch.float16, device="cuda").uniform_(-1, 1))
C = torch.ceil(torch.empty((m, n), dtype=torch.float16, device="cuda").uniform_(-1, 1))
D = torch.zeros_like(C)

# 调用.so文件中的函数
cutlass_kernel(A.data_ptr(), B.data_ptr(), C.data_ptr(), D.data_ptr(), ctypes.c_void_p(0))

# D = torch.matmul(A, B) * 0.2 + C * 0.8
# verify it

D_ref = torch.matmul(A, B) * 0.2 + C * 0.8
# use torch.testing.assert_allclose to compare the results
torch.testing.assert_allclose(D, D_ref, rtol=1e-3, atol=1e-3)