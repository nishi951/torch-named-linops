import marimo

__generated_with = "0.1.67"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    from dataclasses import dataclass
    import numpy as np
    return dataclass, mo, np


@app.cell
def __(dataclass):
    @dataclass
    class BatchedDim:
        name: str
        batch_size: int
    return BatchedDim,


@app.cell
def __(BatchedDim):
    shape1 = (BatchedDim('C', 3), BatchedDim('x', None))
    shape2 = (BatchedDim('C', 3), BatchedDim('x', None))
    shape1 == shape2
    return shape1, shape2


@app.cell
def __(BatchedDim, shape1):
    shape3 = (BatchedDim('C', 4), BatchedDim('x', 2))
    shape3 == shape1
    return shape3,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
