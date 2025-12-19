import numpy as np
from scipy.ndimage import gaussian_filter
import config
from scipy.sparse import lil_matrix
import torch

# Function to generate field coefficients a(x)
def generate_a(
        N=config.DEFAULT_N, 
        sigma=config.DEFAULT_SIGMA, 
        contrast=config.DEFAULT_CONTRAST
):
    g = np.random.randn(N, N)
    g = gaussian_filter(g, sigma=sigma)
    g = contrast * (g - g.mean())
    a = np.exp(g)
    return a

# Function to build the SPD matrix A from coefficients a(x)
def build_A(a):
    N = a.shape[0]
    h2 = (N + 1) ** 2
    A = lil_matrix((N * N, N * N))

    def idx(i, j):
        return i + j * N
    
    for i in range(N):
        for j in range(N):
            k = idx(i, j)
            
            if i < N - 1:
                a_e = 2 * a[i, j] * a[i + 1, j] / (a[i, j] + a[i + 1, j])
                A[k, k] += a_e * h2
                A[k, idx(i + 1, j)] -= a_e * h2

            if i > 0:
                a_w = 2 * a[i, j] * a[i - 1, j] / (a[i, j] + a[i - 1, j])
                A[k, k] += a_w * h2
                A[k, idx(i - 1, j)] -= a_w * h2

            if j < N - 1:
                a_n = 2 * a[i, j] * a[i, j + 1] / (a[i, j] + a[i, j + 1])
                A[k, k] += a_n * h2
                A[k, idx(i, j + 1)] -= a_n * h2

            if j > 0:
                a_s = 2 * a[i, j] * a[i, j - 1] / (a[i, j] + a[i, j - 1])
                A[k, k] += a_s * h2
                A[k, idx(i, j - 1)] -= a_s * h2
    A = A.tocsr()
    A = torch.from_numpy(A.toarray()).float()

    return A

def split_L_components(out, eps=config.eps):
    L_diag = L_diag = torch.exp(out[0]) + config.eps
    L_west = out[1]
    L_south = out[2]

    return L_diag, L_west, L_south

def build_L(A, out, alpha=config.alpha, eps=config.eps):
    _, N, _ = out.shape
    n = N * N
    device = out.device

    L = torch.zeros((n, n), device=device)

    D = torch.diag(A).reshape(N, N)
    L_diag = torch.sqrt(D) * torch.exp(out[0]) + config.eps
    L_west  = alpha * torch.tanh(out[1])
    L_south = alpha * torch.tanh(out[2])

    for j in range(N):
        for i in range(N):
            k = i + j * N

            # diagonal
            L[k, k] = L_diag[j, i]

            # west neighbor
            if i > 0:
                kw = (i - 1) + j * N
                L[k, kw] = L_west[j, i]

            # south neighbor
            if j > 0:
                ks = i + (j - 1) * N
                L[k, ks] = L_south[j, i]

    return L


# Function to apply M inverse using the lower-triangular matrix L constructed by
# our neural network and build_L
def apply_M_inv(L, b):
    b = torch.as_tensor(b, dtype=L.dtype, device=L.device)

    if b.ndim == 1:
        b = b.unsqueeze(1)

    y = torch.linalg.solve_triangular(L, b, upper=False)
    x = torch.linalg.solve_triangular(L.T, y, upper=True)

    return x.squeeze(1)



def loss(A, apply_M_inv, num_probes=config.DEFAULT_NUM_PROBES):
    n = A.shape[0]
    loss = 0.0

    for _ in range(num_probes):
        v = torch.randn(n, device=A.device)
        Av = A @ v
        M_inv_Av = apply_M_inv(Av)
        r = v - M_inv_Av
        loss += torch.dot(r, r)

    return loss / num_probes


def apply_jacobi_inv(A, b):
    D_inv = 1.0 / torch.diag(A)
    return D_inv * b
