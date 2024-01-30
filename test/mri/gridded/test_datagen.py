import marimo

__generated_with = "0.1.88"
app = marimo.App()


@app.cell
def __():
    import marimo as mo

    import numpy as np
    import torch
    import sigpy as sp
    import matplotlib
    import matplotlib.pyplot as plt

    from torchlinops.mri.gridded.backend.datagen import (
        ImplicitGROGDataset,
        ImplicitGROGDatasetConfig,
    )
    return (
        ImplicitGROGDataset,
        ImplicitGROGDatasetConfig,
        matplotlib,
        mo,
        np,
        plt,
        sp,
        torch,
    )


@app.cell
def __(mo):
    mo.md("# Testing Implicit GROG Data Generation")
    return


@app.cell
def __():
    from trj import spiral2d
    from mr_phantoms import shepp_logan

    # Generate phantom
    # Generate spiral trajectory
    # Simulate data
    return shepp_logan, spiral2d


if __name__ == "__main__":
    app.run()
