import pytest
import torch

from torchlinops import Add, Concat, Dense, Stack


@pytest.fixture
def dense_linops():
    A = Dense(torch.randn(1, 4, 3), ("N", "M", "K"), ("N", "K"), ("N", "M"))
    B = Dense(torch.randn(1, 4, 3), ("N", "M", "K"), ("N", "K"), ("N", "M"))
    C = Dense(torch.randn(1, 4, 3), ("N", "M", "K"), ("N", "K"), ("N", "M"))
    return A, B, C


class TestAddThreaded:
    def test_add_threaded(self, dense_linops):
        A, B, C = dense_linops
        add = Add(A, B, C)
        x = torch.randn(1, 3)
        y = add(x)
        assert y.shape == torch.Size([1, 4])

    def test_add_non_threaded(self, dense_linops):
        A, B, C = dense_linops
        add = Add(A, B, C, threaded=False)
        x = torch.randn(1, 3)
        y = add(x)
        assert y.shape == torch.Size([1, 4])

    def test_add_threaded_vs_non_threaded(self, dense_linops):
        A, B, C = dense_linops
        x = torch.randn(1, 3)

        add_threaded = Add(A, B, C)
        add_non_threaded = Add(A, B, C, threaded=False)

        y_threaded = add_threaded(x)
        y_non_threaded = add_non_threaded(x)

        assert torch.allclose(y_threaded, y_non_threaded)

    def test_add_num_workers(self, dense_linops):
        A, B, C = dense_linops
        add = Add(A, B, C, num_workers=2)
        x = torch.randn(1, 3)
        y = add(x)
        assert y.shape == torch.Size([1, 4])


class TestConcatThreaded:
    def test_concat_threaded(self, dense_linops):
        A, B = dense_linops[:2]
        concat = Concat(A, B, idim="N")
        x = torch.randn(2, 3)
        y = concat(x)
        assert y.shape == torch.Size([1, 4])

    def test_concat_non_threaded(self, dense_linops):
        A, B = dense_linops[:2]
        concat = Concat(A, B, idim="N", threaded=False)
        x = torch.randn(2, 3)
        y = concat(x)
        assert y.shape == torch.Size([1, 4])

    def test_concat_threaded_vs_non_threaded(self, dense_linops):
        A, B = dense_linops[:2]
        x = torch.randn(2, 3)

        concat_threaded = Concat(A, B, idim="N")
        concat_non_threaded = Concat(A, B, idim="N", threaded=False)

        y_threaded = concat_threaded(x)
        y_non_threaded = concat_non_threaded(x)

        assert torch.allclose(y_threaded, y_non_threaded)

    def test_concat_num_workers(self, dense_linops):
        A, B = dense_linops[:2]
        concat = Concat(A, B, idim="N", num_workers=2)
        x = torch.randn(2, 3)
        y = concat(x)
        assert y.shape == torch.Size([1, 4])


class TestStackThreaded:
    def test_stack_threaded(self, dense_linops):
        A, B, C = dense_linops
        stack = Stack(A, B, C, odim_and_idx=("L", 0))
        x = torch.randn(1, 3)
        y = stack(x)
        assert y.shape == torch.Size([3, 1, 4])

    def test_stack_non_threaded(self, dense_linops):
        A, B, C = dense_linops
        stack = Stack(A, B, C, odim_and_idx=("L", 0), threaded=False)
        x = torch.randn(1, 3)
        y = stack(x)
        assert y.shape == torch.Size([3, 1, 4])

    def test_stack_threaded_vs_non_threaded(self, dense_linops):
        A, B, C = dense_linops
        x = torch.randn(1, 3)

        stack_threaded = Stack(A, B, C, odim_and_idx=("L", 0))
        stack_non_threaded = Stack(A, B, C, odim_and_idx=("L", 0), threaded=False)

        y_threaded = stack_threaded(x)
        y_non_threaded = stack_non_threaded(x)

        assert torch.allclose(y_threaded, y_non_threaded)

    def test_stack_num_workers(self, dense_linops):
        A, B, C = dense_linops
        stack = Stack(A, B, C, odim_and_idx=("L", 0), num_workers=2)
        x = torch.randn(1, 3)
        y = stack(x)
        assert y.shape == torch.Size([3, 1, 4])
