import pytest
from collections import OrderedDict

from torchlinops import NamedShape, NS, ND


def test_getter_setter():
    shape = NS(('A', 'B'), ('C',))
    assert shape.ishape == ('A', 'B')
    assert shape.oshape == ('C',)

def test_new_batch():
    shape = NS(('A', 'B'), ('C',))
    shape.add('shared_batch', ('A',))
    assert shape.shared_batch == ('A',)

    shape.ishape = ('E', 'F')
    assert shape.shared_batch == ('E',)

def test_new_dict():
    shape = NS(('A', 'B'), ('C',))
    shape.add('axes_lengths', {'A': 2})
    assert shape.axes_lengths[ND.infer('A')] == 2

    shape.ishape = ('E', 'F')
    assert shape.axes_lengths[ND.infer('E')] == 2

def test_diag():
    shape = NS(('A', 'B'))
    assert shape.ishape == ('A', 'B')
    assert shape.oshape == ('A', 'B')
    shape.ishape = ('C', 'D')
    assert shape.oshape == ('C', 'D')

def test_empty_dim():
    shape = NS(tuple(), ("C",))
    adj_shape = shape.H
    #normal_shape = shape.N

    adj_shape.ishape = ("C1",)
    assert shape.oshape == ("C1",)

def test_adjoint():
    shape = NamedShape(("A", "B"), ("C",))
    adj_shape = shape.H
    adj_shape.ishape = ("D",)
    assert shape.oshape == ("D",)

def test_product():
    shape1 = NS(("A", "B"), ("C",))
    shape2 = NS(("E", "F"))

    shape12 = shape1 + shape2
    assert shape12.ishape == ("A", "B", "E", "F")
    assert shape12.oshape == ("C", "E", "F")

    shape21 = shape2 + shape1
    assert shape21.ishape == ("E", "F", "A", "B")
    assert shape21.oshape == ("E", "F", "C")

    adj_shape12 = shape12.H
    adj_shape12.ishape = ("G", "H", "I")
    assert shape12.ishape == ("A", "B", "H", "I")
    assert shape12.oshape == ("G", "H", "I")
