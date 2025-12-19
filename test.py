import config
import tools.tools as tools
import torch
import numpy as np
import scipy
from NeuralPreconditioner import net

# Generate coefficient field
a = tools.generate_a(N=32)
a_t = torch.tensor(np.log(a), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Build diffusion matrix
A = tools.build_A(a)        # shape: (n, n)

# Run CNN
model = net()
out = model(a_t)[0]       # (3, 32, 32)

# Assemble L
L = tools.build_L(A, out)       # (n, n)
diag = torch.diag(L)
print("Nonzero diag entries:", torch.count_nonzero(torch.diag(L)))
print("min diag(L): ")
print("Matrix size:", L.shape[0])

def apply_learned_M_inv(b):
    return tools.apply_M_inv(L, b)

loss_learned = tools.loss(A, apply_learned_M_inv)
loss_jacobi  = tools.loss(A, lambda b: tools.apply_jacobi_inv(A, b))

print("Learned M loss:", loss_learned.item())
print("Jacobi loss:", loss_jacobi.item())
