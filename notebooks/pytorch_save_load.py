import marimo

__generated_with = "0.1.67"
app = marimo.App()


@app.cell
def __():
    import marimo as mo

    import torch
    import torch.nn as nn
    return mo, nn, torch


@app.cell
def __(mo, nn, torch):
    class BigBoi(nn.Module):
        def __init__(self, chonk: torch.Tensor):
            super().__init__()
            self.chonk = nn.Parameter(chonk, requires_grad=False)

        def __getitem__(self, idx):
            return BigBoi(self.chonk[idx])

    mo.md("""
    We initialize an object that contains a single parameter.
    """)
    return BigBoi,


@app.cell
def __(BigBoi, mo, torch):
    chonk = torch.zeros(10, 13)
    obj1 = BigBoi(chonk)
    obj2 = obj1[:2, :]
    print(obj1.chonk)
    mo.md("""
    We start by setting all the parameters in the module to 0.
    """)

    return chonk, obj1, obj2


@app.cell
def __(mo, obj1, obj2, torch):
    state_dict = obj2.state_dict()
    state_dict['chonk'] = torch.ones_like(state_dict['chonk'])
    obj2.load_state_dict(state_dict)
    print(obj1.chonk)
    mo.md("""
    We then modify the state dict of the derived object and observe
    that the changes propagate to the original object.
    """)
    return state_dict,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
