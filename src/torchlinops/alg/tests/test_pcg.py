import torch

from torchlinops.alg.pcg import CGRun, conjugate_gradients


def test_conjugate_gradients_simple():
    n = 10
    A = torch.eye(n) * 2.0
    y = torch.ones(n)

    def A_op(x):
        return A @ x

    x = conjugate_gradients(A_op, y, max_num_iters=20, disable_tracking=True)

    assert x is not None
    assert x.shape == y.shape


def test_conjugate_gradients_with_x0():
    n = 10
    A = torch.eye(n) * 2.0
    y = torch.ones(n)
    x0 = torch.zeros(n)

    def A_op(x):
        return A @ x

    x = conjugate_gradients(A_op, y, x0=x0, max_num_iters=20, disable_tracking=True)

    assert x.shape == y.shape


def test_conjugate_gradients_max_iters():
    n = 10
    A = torch.eye(n) * 2.0
    y = torch.ones(n)

    def A_op(x):
        return A @ x

    x = conjugate_gradients(A_op, y, max_num_iters=5, disable_tracking=True)

    assert x is not None


def test_conjugate_gradients_disable_tracking():
    n = 10
    A = torch.eye(n) * 2.0
    y = torch.ones(n)

    def A_op(x):
        return A @ x

    x = conjugate_gradients(A_op, y, disable_tracking=True, max_num_iters=20)

    assert x is not None


def test_conjugate_gradients_tracking():
    n = 10
    A = torch.eye(n) * 2.0
    y = torch.ones(n)

    def A_op(x):
        return A @ x

    x = conjugate_gradients(A_op, y, disable_tracking=False, max_num_iters=20)

    assert x is not None


def test_cgrun_update():
    n = 10
    A = torch.eye(n) * 2.0
    y = torch.ones(n)

    def A_op(x):
        return A @ x

    run = CGRun(ltol=1e-5, gtol=1e-3, A=A_op, y=y, disable=False)
    x = torch.ones(n)
    run.update(x)

    assert run.x_out is not None
    assert run.loss < float("inf")


def test_cgrun_update_disabled():
    n = 10
    A = torch.eye(n) * 2.0
    y = torch.ones(n)

    def A_op(x):
        return A @ x

    run = CGRun(ltol=1e-5, gtol=1e-3, A=A_op, y=y, disable=True)
    x = torch.ones(n)
    run.update(x)

    assert run.x_out is not None
    assert run.loss == float("inf")


def test_cgrun_is_not_converged_initially():
    n = 10
    A = torch.eye(n) * 100.0
    y = torch.ones(n)

    def A_op(x):
        return A @ x

    run = CGRun(ltol=1e-5, gtol=1e-3, A=A_op, y=y, disable=False)
    x = torch.ones(n) * 0.01
    run.update(x)

    assert not run.is_converged()


def test_cgrun_is_converged_at_solution():
    """CGRun.is_converged() should return True once both criteria are met."""
    n = 5
    A_mat = torch.eye(n) * 2.0
    y = torch.ones(n)

    def A_op(x):
        return A_mat @ x

    # Exact solution: x = y / 2
    x_exact = y / 2.0
    run = CGRun(ltol=1.0, gtol=1.0, A=A_op, y=y, disable=False)
    run.update(x_exact)
    run.update(x_exact)  # Two updates so prev_loss is set
    assert run.is_converged()


def test_cgrun_set_postfix_disabled_skips():
    """CGRun.set_postfix should return early when disable=True."""
    n = 5
    A_mat = torch.eye(n) * 2.0
    y = torch.ones(n)

    def A_op(x):
        return A_mat @ x

    run = CGRun(ltol=1e-5, gtol=1e-3, A=A_op, y=y, disable=True)

    class FakePbar:
        called = False

        def set_postfix(self, *a, **kw):
            FakePbar.called = True

    pbar = FakePbar()
    run.set_postfix(pbar)
    assert not pbar.called
