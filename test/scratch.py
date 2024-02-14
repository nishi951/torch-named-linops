import marimo

__generated_with = "0.2.2"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import numpy as np
    from einops import rearrange
    return mo, np, rearrange


@app.cell
def __(np, rearrange):
    cal_size = (32, 64, 64)
    coords = tuple((np.arange(im_size) - im_size // 2) for im_size in cal_size)
    xyz = np.meshgrid(*coords)
    xyz = np.stack(xyz, axis=-1)
    xyz = rearrange(xyz, '... d -> (...) d')
    i = np.random.randint(xyz.shape[0], size=3)
    print(xyz[i])
    return cal_size, coords, i, xyz


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
