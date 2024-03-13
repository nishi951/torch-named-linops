import marimo

__generated_with = "0.2.4"
app = marimo.App()


@app.cell
def __():
    import marimo as mo

    import numpy as np
    import matplotlib.pyplot as plt

    import brainweb

    return brainweb, mo, np, plt


@app.cell
def __(brainweb):
    data = brainweb.brainweb_phantom()
    return (data,)


@app.cell
def __(data, mo):
    img_key = mo.ui.dropdown(
        options=list(data.keys()), label="Image to show", value="t1w"
    )
    img_key

    return (img_key,)


@app.cell
def __(data, img_key, mo):
    xmax, ymax, zmax = data[img_key.value].shape
    x = mo.ui.slider(start=0, stop=xmax - 1, label="x", value=xmax // 2)
    y = mo.ui.slider(start=0, stop=ymax - 1, label="y", value=ymax // 2)
    z = mo.ui.slider(start=0, stop=zmax - 1, label="z", value=zmax // 2)
    xyz = mo.vstack([x, y, z])
    xyz
    return x, xmax, xyz, y, ymax, z, zmax


@app.cell
def __(data, img_key, np, x, y, z):
    from einops import rearrange

    img = data[img_key.value]
    img = rearrange(img, "z y x -> x y z")
    img = np.flip(img, axis=(1, 2, 0))
    axial = img[:, :, z.value]
    sagittal = img[
        :,
        y.value,
    ]
    coronal = img[x.value, :, :]
    return axial, coronal, img, rearrange, sagittal


@app.cell
def __(axial, coronal, np, plt, sagittal):
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(np.rot90(np.abs(axial)))
    ax[0].set_title("axial")
    ax[1].imshow(np.rot90(np.abs(sagittal)))
    ax[1].set_title("sagittal")
    ax[2].imshow(np.rot90(np.abs(coronal)))
    ax[2].set_title("coronal")
    return ax, fig


@app.cell
def __():
    import sigpy as sp
    import sigpy.mri as mri

    return mri, sp


@app.cell
def __(img, mri, x, y, z):
    mps = mri.birdcage_maps((20, *img.shape))
    axial_mps = mps[0, :, :, z.value]
    sagittal_mps = mps[0, :, y.value, :]
    coronal_mps = mps[0, x.value, :, :]
    return axial_mps, coronal_mps, mps, sagittal_mps


@app.cell
def __(axial_mps, coronal_mps, np, plt, sagittal_mps):
    fig2, ax2 = plt.subplots(nrows=1, ncols=3)
    ax2[0].imshow(np.rot90(np.abs(axial_mps)))
    ax2[1].imshow(np.rot90(np.abs(sagittal_mps)))
    ax2[2].imshow(np.rot90(np.abs(coronal_mps)))
    return ax2, fig2


@app.cell
def __(mo):
    mo.md("## T1/T2 distributions")
    return


@app.cell
def __(data, np, plt):
    T1, T2 = data["t1"].reshape(-1), data["t2"].reshape(-1)
    counts, bins = np.histogram(T1)
    plt.hist(bins[:-1], bins, weights=counts)
    return T1, T2, bins, counts


@app.cell
def __(T2, np, plt):
    counts2, bins2 = np.histogram(T2)
    plt.hist(bins2[:-1], bins2, weights=counts2)
    return bins2, counts2


@app.cell
def __(np):
    T1x = np.concatenate((np.array(range(20, 301, 20)), np.array(range(340, 821, 40))))
    T2x = np.concatenate((np.array(range(10, 51, 5)), np.array(range(60, 101, 10))))

    return T1x, T2x


@app.cell
def __(T1x, T2x):
    print(T1x, T2x)
    return


@app.cell
def __(T1x, T2x, np):
    t1t2pd = np.stack(np.meshgrid(T1x, T2x, 1.0, indexing="ij"), axis=0)
    return (t1t2pd,)


@app.cell
def __(t1t2pd):
    t1t2pd.shape
    return


@app.cell
def __(T1x):
    print(len(T1x))
    return


@app.cell
def __(t1t2pd):
    tuple(t1t2pd)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
