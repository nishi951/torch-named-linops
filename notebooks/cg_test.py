import marimo

__generated_with = "0.1.73"
app = marimo.App()


@app.cell
def __(mo):
    mo.md("""# MRF Reconstruction""")
    return


@app.cell
def __(mo):
    mo.md(
        """
        Enter the path to some data below
        """
    )
    return


@app.cell
def __(mo):
    # form = mo.vstack([mo.ui.file(kind='button'), mo.ui.file(kind='area'), mo.ui.text(kind='text', placeholder='File Path')])
    data_path = mo.ui.text(placeholder="File Path").form()
    data_path
    return (data_path,)


@app.cell
def __(mo):
    mo.md("### Conjugate Gradient Recon")
    return


@app.cell
def __(MRFSubspData, data_path):
    data = MRFSubspData.loadnp(data_path.value)
    return (data,)


@app.cell
def __(ConjugateGradient, make_linop):
    def run_cg(data):
        A = make_linop(data.trj, data.mps, data.sqrt_dcf)
        AHb = A.H(data.ksp)
        cg = ConjugateGradient(A.N, A.H(data.ksp))
        recon = cg(AHb, AHb)

    return (run_cg,)


@app.cell
def __():
    import marimo as mo

    import torch

    from torchlinops.core import Diagonal, Dense
    from torchlinops.mri.linops import NUFFT, SENSE
    from dataio import MRFSubspData

    return Dense, Diagonal, MRFSubspData, NUFFT, SENSE, mo, torch


@app.cell
def __(Dense, Diagonal, NUFFT, SENSE, data):
    def make_linop(
        trj,
        mps,
        sqrt_dcf,
    ):
        S = SENSE(data.mps)
        F = NUFFT(
            data.trj,
        )
        D = Diagonal(data.sqrt_dcf)
        Phi = Dense(data.phi)
        A = D @ F @ S @ Phi
        return A

    return (make_linop,)


if __name__ == "__main__":
    app.run()
