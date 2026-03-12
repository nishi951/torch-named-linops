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
        add = Add(A, B, C)
        add.threaded = False
        x = torch.randn(1, 3)
        y = add(x)
        assert y.shape == torch.Size([1, 4])

    def test_add_threaded_vs_non_threaded(self, dense_linops):
        A, B, C = dense_linops
        x = torch.randn(1, 3)

        add_threaded = Add(A, B, C)
        add_non_threaded = Add(A, B, C)
        add_non_threaded.threaded = False

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


class TestSharedLinopCopying:
    """Tests for automatic shallow copying of shared linops in Threadable classes."""

    def test_add_shared_linop_copies(self):
        """Shared linop in Add should be copied."""
        A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
        add = Add(A, A)

        assert add.linops[0] is not A
        assert add.linops[0] is not add.linops[1]
        assert add.linops[0].weight.data_ptr() == add.linops[1].weight.data_ptr()

    def test_concat_shared_linop_copies(self):
        """Shared linop in Concat should be copied."""
        A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
        concat = Concat(A, A, idim="K")

        assert concat.linops[0] is not A
        assert concat.linops[0] is not concat.linops[1]
        assert concat.linops[0].weight.data_ptr() == concat.linops[1].weight.data_ptr()

    def test_stack_shared_linop_copies(self):
        """Shared linop in Stack should be copied."""
        A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
        stack = Stack(A, A, odim_and_idx=("L", 0))

        assert stack.linops[0] is not A
        assert stack.linops[0] is not stack.linops[1]
        assert stack.linops[0].weight.data_ptr() == stack.linops[1].weight.data_ptr()

    def test_nested_shared_linop_copies(self):
        """Shared linop nested in Chains inside Concat should be copied."""
        from torchlinops import Chain

        A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
        B = Dense(torch.randn(4, 4), ("N", "M"), ("M",), ("N",))

        chain1 = Chain(A, B)
        chain2 = Chain(A, B)
        concat = Concat(chain1, chain2, idim="K")

        assert concat.linops[0] is not concat.linops[1]
        assert concat.linops[0].linops[0] is not concat.linops[1].linops[0]

    def test_nested_shared_linop_shallow_copy(self):
        """Nested shared linops should still share tensor storage."""
        from torchlinops import Chain

        A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
        B = Dense(torch.randn(4, 4), ("N", "M"), ("M",), ("N",))

        chain1 = Chain(A, B)
        chain2 = Chain(A, B)
        concat = Concat(chain1, chain2, idim="K")

        assert (
            concat.linops[0].linops[0].weight.data_ptr()
            == concat.linops[1].linops[0].weight.data_ptr()
        )

    def test_threading_preserved_after_copy(self):
        """Threading should still work after copying shared linops."""
        A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
        concat = Concat(A, A, idim="K", threaded=True)

        assert concat.threaded is True

        x = torch.randn(6)
        y = concat(x)
        assert y.shape == (4,)

    def test_add_input_listener_independent(self):
        """Copied linops should have independent input_listener references."""
        A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
        add = Add(A, A)

        # Check that _input_listener forwards to the correct parent
        assert add.linops[0]._input_listener._obj is add
        assert add.linops[0]._input_listener._attr == "input_listener"
        assert add.linops[1]._input_listener._obj is add
        assert add.linops[1]._input_listener._attr == "input_listener"
        # The two linops objects should be different
        assert add.linops[0] is not add.linops[1]

    def test_concat_input_listener_independent(self):
        """Copied linops should have independent input_listener references."""
        A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
        concat = Concat(A, A, idim="K")

        assert concat[0]._input_listener._obj is concat
        assert concat[0]._input_listener._attr == "input_listener"
        assert concat[1]._input_listener._obj is concat
        assert concat[1]._input_listener._attr == "input_listener"

    def test_nested_input_listener_independent(self):
        """Nested copied linops should have independent input_listener references."""
        from torchlinops import Chain

        A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
        B = Dense(torch.randn(4, 4), ("N", "M"), ("M",), ("N",))

        chain1 = Chain(A, B)
        chain2 = Chain(A, B)
        concat = Concat(chain1, chain2, idim="K")

        assert concat.linops[0]._input_listener._obj is concat
        assert concat.linops[1]._input_listener._obj is concat

        assert concat.linops[0].linops[0] is not concat.linops[1].linops[0]
        assert concat.linops[0].linops[0]._input_listener._obj is concat.linops[0]
        assert concat.linops[1].linops[0]._input_listener._obj is concat.linops[1]
