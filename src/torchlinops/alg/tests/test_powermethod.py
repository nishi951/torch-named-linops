import torch

from torchlinops.alg.powermethod import power_method


def test_power_method_simple():
    n = 10
    A = torch.eye(n) * 2.0
    A_op = lambda x: A @ x

    v_init = torch.randn(n)
    v, eigenvalue = power_method(A_op, v_init, max_iters=50)

    assert eigenvalue.shape == ()
    assert abs(eigenvalue.item() - 2.0) < 0.1


def test_power_method_different_eigenvalues():
    n = 5
    diag = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    A = torch.diag(diag)
    A_op = lambda x: A @ x

    v_init = torch.ones(n)
    v, eigenvalue = power_method(A_op, v_init, max_iters=100)

    assert abs(eigenvalue.item() - 5.0) < 0.1


def test_power_method_custom_tol():
    n = 5
    A = torch.eye(n) * 3.0
    A_op = lambda x: A @ x

    v_init = torch.randn(n)
    v, eigenvalue = power_method(A_op, v_init, max_iters=10, tol=1e-3)

    assert eigenvalue.shape == ()


def test_power_method_with_tqdm():
    n = 3
    A = torch.eye(n) * 2.0
    A_op = lambda x: A @ x

    v_init = torch.randn(n)
    v, eigenvalue = power_method(
        A_op, v_init, max_iters=5, tqdm_kwargs={"disable": True}
    )

    assert eigenvalue.shape == ()


def test_power_method_complex():
    n = 3
    A = torch.eye(n) * (1 + 0j)
    A_op = lambda x: A @ x

    v_init = torch.randn(n, dtype=torch.complex64)
    v, eigenvalue = power_method(A_op, v_init, max_iters=10)

    assert eigenvalue.shape == ()


def test_power_method_batch():
    n = 4
    batch = 3
    A = torch.stack([torch.eye(n) * (i + 1) for i in range(batch)])

    def batched_A(x):
        return torch.einsum("bij,bj->bi", A, x)

    v_init = torch.randn(batch, n)
    v, eigenvalue = power_method(batched_A, v_init, max_iters=50, dim=1)

    assert eigenvalue.shape == (batch,)
