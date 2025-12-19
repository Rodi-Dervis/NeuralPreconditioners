import tools.tools as tools
import torch
import numpy as np
from NeuralPreconditioner import net

# Generate coefficient field
a = tools.generate_a(N=32)
a_t = torch.tensor(np.log(a), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Build diffusion matrix
A = tools.build_A(a)

# Run CNN
model = net()
out = model(a_t)[0]

# Assemble L
L = tools.build_L(A, out)
diag = torch.diag(L)

def apply_learned_M_inv(b):
    return tools.apply_M_inv(L, b)

# Calculate loss
loss_learned = tools.loss(A, apply_learned_M_inv)
loss_jacobi  = tools.loss(A, lambda b: tools.apply_jacobi_inv(A, b))

print("Learned M loss:", loss_learned.item())
print("Jacobi loss:", loss_jacobi.item())
