"""
Samplers for the orthogonal O(N) and special orthogonal SO(N) groups
modeled after the functions in scipy.stats
 - special_ortho_group.rvs(dim) and
 - ortho_group.rvs(dim).
"""
import torch


def ortho_group_rvs(dim, size=1, output_numpy=False,
                    device='cuda:0', dtype=torch.float32):
    """
    Draw random samples from O(N).

    Parameters
    ----------
    dim : integer
        Dimension of rotation space (N).
    size : integer, optional
        Number of samples to draw (default 1).
    output_numpy : bool, optional
        If True, returns a numpy array, default is to return a torch Tensor.
    device : pytorch device
        Where to do the computation.
    dtype : pytorch dtype
        pytorch dtype, impact precision.

    Returns
    -------
    rvs : ndarray or torch Tensor
        Random size N-dimensional matrices, dimension (size, dim, dim)

    """

    size = int(size)
    if size > 1:
        return torch.stack([ortho_group_rvs(dim, size=1, device=device, dtype=dtype)
                            for i in range(size)])

    H = torch.eye(dim, device=device, dtype=dtype)
    for n in range(dim):
        x = torch.randn(dim-n, device=device, dtype=dtype)
        norm2 = torch.dot(x, x)
        x0 = x[0].clone()
        # random sign, 50/50, but chosen carefully to avoid roundoff error
        D = torch.sign(x[0]) if x[0] != 0 else 1
        x[0] = x[0] + (D * torch.sqrt(norm2))
        x /= torch.sqrt((norm2 - x0**2 + x[0]**2) / 2.)
        # Householder transformation
        x_un = torch.unsqueeze(x, 0)
        Hx = -D*(torch.eye(dim-n, device=device, dtype=dtype) - x_un * x_un.t())
        mat = torch.eye(dim, device=device, dtype=dtype)
        mat[n:, n:] = Hx
        H = torch.mm(H, mat)
    if output_numpy:
        H = H.detach().cpu().numpy()
    return H


def special_ortho_group_rvs(dim, size=1, device='cuda:0', dtype=torch.float32,
                            output_numpy=False):
    """
    Draw random samples from SO(N).

    Parameters
    ----------
    dim : integer
        Dimension of rotation space (N).
    size : integer, optional
        Number of samples to draw (default 1).
    output_numpy : bool, optional
        If True, returns a numpy array, default is to return a torch Tensor.
    device : pytorch device
        Where to do the computation.
    dtype : pytorch dtype
        pytorch dtype, impact precision.

    Returns
    -------
    rvs : ndarray or torch Tensor
        Random size N-dimensional matrices, dimension (size, dim, dim)

    """

    size = int(size)
    if size > 1:
        return torch.stack([special_ortho_group_rvs(dim, size=1,
                                                    device=device, dtype=dtype)
                            for i in range(size)])

    H = torch.eye(dim, device=device, dtype=dtype)
    D = torch.empty(dim, device=device, dtype=dtype)
    for n in range(dim-1):
        x = torch.randn(dim-n, device=device, dtype=dtype)
        norm2 = torch.dot(x, x)
        x0 = x[0].clone()
        D[n] = torch.sign(x[0]) if x[0] != 0 else 1
        x[0] += D[n] * torch.sqrt(norm2)
        x /= torch.sqrt((norm2 - x0**2 + x[0]**2) / 2.)
        # Householder transformation
        x_un = torch.unsqueeze(x, 0)
        Hx = torch.eye(dim-n, device=device, dtype=dtype) - x_un * x_un.t()
        mat = torch.eye(dim, device=device, dtype=dtype)
        mat[n:, n:] = Hx
        H = torch.mm(H, mat)
    D[-1] = (-1)**(dim - 1) * D[:-1].prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.t()).t()
    if output_numpy:
        H = H.detach().cpu().numpy()
    return H
