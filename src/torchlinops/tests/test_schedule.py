# import pytest
# import torch

# from torchlinops import Add, Chain, Concat, Dense, Stack
# from torchlinops.linops.schedule import ExecutionSchedule, topological_groups


# @pytest.fixture
# def dense_linops():
#     A = Dense(torch.randn(1, 4, 3), ("N", "M", "K"), ("N", "K"), ("N", "M"))
#     B = Dense(torch.randn(1, 4, 3), ("N", "M", "K"), ("N", "K"), ("N", "M"))
#     C = Dense(torch.randn(1, 4, 3), ("N", "M", "K"), ("N", "K"), ("N", "M"))
#     return A, B, C


# class TestAddThreaded:
#     def test_add_threaded(self, dense_linops):
#         A, B, C = dense_linops
#         add = Add(A, B, C)
#         x = torch.randn(1, 3)
#         y = add(x)
#         assert y.shape == torch.Size([1, 4])

#     def test_add_non_threaded(self, dense_linops):
#         A, B, C = dense_linops
#         add = Add(A, B, C)
#         add.threaded = False
#         x = torch.randn(1, 3)
#         y = add(x)
#         assert y.shape == torch.Size([1, 4])

#     def test_add_threaded_vs_non_threaded(self, dense_linops):
#         A, B, C = dense_linops
#         x = torch.randn(1, 3)

#         add_threaded = Add(A, B, C)
#         add_non_threaded = Add(A, B, C)
#         add_non_threaded.threaded = False

#         y_threaded = add_threaded(x)
#         y_non_threaded = add_non_threaded(x)

#         assert torch.allclose(y_threaded, y_non_threaded)

#     def test_add_num_workers(self, dense_linops):
#         A, B, C = dense_linops
#         add = Add(A, B, C)
#         add.num_workers = 2
#         x = torch.randn(1, 3)
#         y = add(x)
#         assert y.shape == torch.Size([1, 4])


# class TestConcatThreaded:
#     def test_concat_threaded(self, dense_linops):
#         A, B = dense_linops[:2]
#         concat = Concat(A, B, idim="N")
#         x = torch.randn(2, 3)
#         y = concat(x)
#         assert y.shape == torch.Size([1, 4])

#     def test_concat_non_threaded(self, dense_linops):
#         A, B = dense_linops[:2]
#         concat = Concat(A, B, idim="N")
#         concat.threaded = False
#         x = torch.randn(2, 3)
#         y = concat(x)
#         assert y.shape == torch.Size([1, 4])

#     def test_concat_threaded_vs_non_threaded(self, dense_linops):
#         A, B = dense_linops[:2]
#         x = torch.randn(2, 3)

#         concat_threaded = Concat(A, B, idim="N")
#         concat_non_threaded = Concat(A, B, idim="N")
#         concat_non_threaded.threaded = False

#         y_threaded = concat_threaded(x)
#         y_non_threaded = concat_non_threaded(x)

#         assert torch.allclose(y_threaded, y_non_threaded)

#     def test_concat_num_workers(self, dense_linops):
#         A, B = dense_linops[:2]
#         concat = Concat(A, B, idim="N")
#         concat.num_workers = 2
#         x = torch.randn(2, 3)
#         y = concat(x)
#         assert y.shape == torch.Size([1, 4])


# class TestStackThreaded:
#     def test_stack_threaded(self, dense_linops):
#         A, B, C = dense_linops
#         stack = Stack(A, B, C, odim_and_idx=("L", 0))
#         x = torch.randn(1, 3)
#         y = stack(x)
#         assert y.shape == torch.Size([3, 1, 4])

#     def test_stack_non_threaded(self, dense_linops):
#         A, B, C = dense_linops
#         stack = Stack(A, B, C, odim_and_idx=("L", 0))
#         stack.threaded = False
#         x = torch.randn(1, 3)
#         y = stack(x)
#         assert y.shape == torch.Size([3, 1, 4])

#     def test_stack_threaded_vs_non_threaded(self, dense_linops):
#         A, B, C = dense_linops
#         x = torch.randn(1, 3)

#         stack_threaded = Stack(A, B, C, odim_and_idx=("L", 0))
#         stack_non_threaded = Stack(A, B, C, odim_and_idx=("L", 0))
#         stack_non_threaded.threaded = False

#         y_threaded = stack_threaded(x)
#         y_non_threaded = stack_non_threaded(x)

#         assert torch.allclose(y_threaded, y_non_threaded)

#     def test_stack_num_workers(self, dense_linops):
#         A, B, C = dense_linops
#         stack = Stack(A, B, C, odim_and_idx=("L", 0))
#         stack.num_workers = 2
#         x = torch.randn(1, 3)
#         y = stack(x)
#         assert y.shape == torch.Size([3, 1, 4])


# class TestSharedLinopIdentity:
#     """Tests that shared linops are NOT copied — identity is preserved."""

#     def test_add_shared_linop_not_copied(self):
#         """Shared linop in Add should be the same object."""
#         A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
#         add = Add(A, A)

#         assert add[0] is A
#         assert add[0] is add[1]
#         assert add[0].weight.data_ptr() == add[1].weight.data_ptr()

#     def test_concat_shared_linop_not_copied(self):
#         """Shared linop in Concat should be the same object."""
#         A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
#         concat = Concat(A, A, idim="K")

#         assert concat[0] is A
#         assert concat[0] is concat[1]
#         assert concat[0].weight.data_ptr() == concat[1].weight.data_ptr()

#     def test_stack_shared_linop_not_copied(self):
#         """Shared linop in Stack should be the same object."""
#         A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
#         stack = Stack(A, A, odim_and_idx=("L", 0))

#         assert stack[0] is A
#         assert stack[0] is stack[1]
#         assert stack[0].weight.data_ptr() == stack[1].weight.data_ptr()

#     def test_nested_shared_linop_not_copied(self):
#         """Shared linops nested in Chains inside Concat should preserve identity."""
#         A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
#         B = Dense(torch.randn(4, 4), ("N", "M"), ("M",), ("N",))

#         chain1 = Chain(A, B)
#         chain2 = Chain(A, B)
#         concat = Concat(chain1, chain2, idim="K")

#         assert concat[0] is not concat[1]
#         # A and B are shared between the two chains
#         assert concat[0][0] is concat[1][0]
#         assert concat[0][1] is concat[1][1]

#     def test_threading_works_with_shared_linops(self):
#         """Threading should still work with shared linops."""
#         A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
#         concat = Concat(A, A, idim="K")
#         concat.threaded = True

#         assert concat.threaded is True

#         x = torch.randn(6)
#         y = concat(x)
#         assert y.shape == (4,)


# class TestExecutionSchedule:
#     """Tests for ExecutionSchedule structure."""

#     def test_add_schedule_is_parallel(self):
#         A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
#         add = Add(A, A)
#         assert add._schedule.is_parallel
#         assert not add._schedule.is_sequential

#     def test_chain_schedule_is_sequential(self):
#         A = Dense(torch.randn(4, 4), ("M", "M"), ("M",), ("M",))
#         chain = Chain(A, A)
#         assert chain._schedule.is_sequential
#         assert not chain._schedule.is_parallel

#     def test_concat_schedule_is_parallel(self):
#         A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
#         concat = Concat(A, A, idim="K")
#         assert concat._schedule.is_parallel

#     def test_stack_schedule_is_parallel(self):
#         A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
#         stack = Stack(A, A, odim_and_idx=("L", 0))
#         assert stack._schedule.is_parallel

#     def test_schedule_to_dict(self):
#         schedule = ExecutionSchedule({0: [], 1: [(0, "end_event")]})
#         d = schedule.to_dict()
#         assert d == {"0": [], "1": [(0, "end_event")]}


# class TestTopologicalGroups:
#     """Tests for topological_groups function."""

#     def test_all_parallel(self):
#         groups = topological_groups({0: [], 1: [], 2: []})
#         assert groups == [[0, 1, 2]]

#     def test_all_sequential(self):
#         groups = topological_groups(
#             {
#                 0: [],
#                 1: [(0, "end_event")],
#                 2: [(1, "end_event")],
#             }
#         )
#         assert groups == [[0], [1], [2]]

#     def test_mixed(self):
#         groups = topological_groups(
#             {
#                 0: [],
#                 1: [],
#                 2: [(0, "end_event")],
#             }
#         )
#         assert groups == [[0, 1], [2]]

#     def test_empty(self):
#         groups = topological_groups({})
#         assert groups == []


# class TestRapidSuccessiveCalls:
#     """Tests that rapid successive calls don't cause issues."""

#     def test_rapid_add_calls(self):
#         A = Dense(torch.randn(4, 3), ("M", "K"), ("K",), ("M",))
#         add = Add(A, A)
#         x1 = torch.randn(3)
#         x2 = torch.randn(3)
#         y1 = add(x1)
#         y2 = add(x2)
#         assert y1.shape == y2.shape == (4,)

#     def test_rapid_chain_calls(self):
#         A = Dense(torch.randn(4, 4), ("M", "M"), ("M",), ("M",))
#         chain = Chain(A, A)
#         x1 = torch.randn(4)
#         x2 = torch.randn(4)
#         y1 = chain(x1)
#         y2 = chain(x2)
#         assert y1.shape == y2.shape == (4,)
