import marimo as mo

app = mo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # MRI Reconstruction

    This guide demonstrates how to perform MRI reconstruction using torchlinops with three key components:

    - **NUFFT**: Non-uniform Fast Fourier Transform for radial sampling
    - **Dense operator**: Multi-coil sensitivity encoding
    - **Diagonal operator**: Density compensation for radial trajectories

    ## Problem Formulation

    The MRI forward model can be written as:

    $$y = E F S x + n$$

    Where:

    - $S$: coil sensitivities (Dense operator)
    - $F$: NUFFT (non-uniform FFT)
    - $E$: density compensation (Diagonal operator)
    - $x$: image to reconstruct
    - $y$: acquired k-space data
    - $n$: noise

    The reconstruction solves the least-squares problem:

    $$\\min_x \\|A x - y\\|_2^2$$

    where $A = E F S$ is the combined forward operator.
    """)
    return


@app.cell
def _(mo):
    mo.md("## Setup and Imports")
    return


@app.cell
def _():
    import sys
    import warnings

    import matplotlib.pyplot as plt
    import torch

    from torchlinops import NUFFT, Dense, Diagonal, Dim
    from torchlinops.alg import conjugate_gradients, power_method

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Ignore warnings
    warnings.filterwarnings("ignore", message=r".*FigureCanvasAgg.*")

    # Reconstruction parameters
    image_size = (128, 128)  # 2D image dimensions
    num_coils = 8  # Number of receiver coils
    num_spokes = 37  # Number of radial spokes
    num_readouts = 256  # Readout points per spoke
    noise_level = 1e-5  # Additive noise level
    use_cartesian = False  # Use Cartesian trajectory for debugging
    return (
        Dense,
        Diagonal,
        Dim,
        NUFFT,
        conjugate_gradients,
        image_size,
        noise_level,
        num_coils,
        num_readouts,
        num_spokes,
        plt,
        power_method,
        sys,
        torch,
        use_cartesian,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Helper Functions

    We define the helper functions here for generating trajectories, sensitivities, and density compensation weights.
    """)
    return


@app.cell
def _(torch):
    def generate_radial_trajectory(num_spokes, num_readouts, grid_size):
        """Generate radial k-space trajectory.

        Parameters
        ----------
        num_spokes : int
            Number of radial spokes
        num_readouts : int
            Number of readout points per spoke
        grid_size : tuple
            Image grid size (Nx, Ny)

        Returns
        -------
        torch.Tensor
            K-space locations with shape (num_spokes * num_readouts, 2)
        """
        Nx, Ny = grid_size
        max_radius = min(Nx, Ny) // 2
        angles = torch.linspace(0, 2 * torch.pi, num_spokes + 1)[:-1]
        readout_pos = torch.linspace(-max_radius, max_radius, num_readouts)
        locs = torch.zeros(num_spokes * num_readouts, 2)
        for _i, angle in enumerate(angles):
            start_idx = _i * num_readouts
            end_idx = (_i + 1) * num_readouts
            x = readout_pos * torch.cos(angle)
            y = readout_pos * torch.sin(angle)
            locs[start_idx:end_idx, 0] = x
            locs[start_idx:end_idx, 1] = y
        locs = locs / (2 * max_radius)
        return locs

    def generate_cartesian_trajectory(grid_size, oversamp=2.0):
        """Generate Cartesian k-space trajectory.

        Parameters
        ----------
        grid_size : tuple
            Image grid size (Nx, Ny)

        Returns
        -------
        torch.Tensor
            K-space locations with shape (Nx * Ny, 2)
        """
        Nx, Ny = grid_size
        x = torch.linspace(-0.5, 0.5, int(Nx * oversamp))
        y = torch.linspace(-0.5, 0.5, int(Ny * oversamp))
        X, Y = torch.meshgrid(x, y, indexing="ij")
        locs = torch.stack([X.flatten(), Y.flatten()], dim=1)
        return locs

    def generate_gaussian_coil_sensitivities(grid_size, num_coils):
        """Generate Gaussian coil sensitivity maps.

        Parameters
        ----------
        grid_size : tuple
            Image grid size (Nx, Ny)
        num_coils : int
            Number of receiver coils

        Returns
        -------
        torch.Tensor
            Coil sensitivity maps with shape (num_coils, Nx, Ny)
        """
        Nx, Ny = grid_size
        x = torch.linspace(-0.5, 0.5, Nx)
        y = torch.linspace(-0.5, 0.5, Ny)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        coil_sens = torch.zeros(num_coils, Nx, Ny, dtype=torch.complex64)
        for c in range(num_coils):
            angle = 2 * torch.pi * c / num_coils
            center_x = 0.3 * torch.cos(torch.tensor(angle))
            center_y = 0.3 * torch.sin(torch.tensor(angle))
            sigma = 1.0
            gaussian = torch.exp(
                -((X - center_x) ** 2 + (Y - center_y) ** 2) / (2 * sigma**2)
            )
            phase = torch.exp(1j * 2 * torch.pi * (0.1 * X + 0.1 * Y))
            coil_sens[c] = gaussian * phase
        return coil_sens

    def analytic_radial_dcf(locs):
        """Compute analytic density compensation for radial trajectories.

        Parameters
        ----------
        locs : torch.Tensor
            K-space locations with shape (N, 2)

        Returns
        -------
        torch.Tensor
            Density compensation weights with shape (N,)
        """
        radius = torch.sqrt(locs[:, 0] ** 2 + locs[:, 1] ** 2)
        dcf = radius + 1e-06
        dcf = dcf / torch.mean(dcf)
        return dcf

    def analytic_cartesian_dcf(locs):
        """Compute analytic density compensation for Cartesian trajectory.

        Parameters
        ----------
        locs : torch.Tensor
            K-space locations

        Returns
        -------
        torch.Tensor
            Density compensation weights (all ones)
        """
        return torch.ones(locs.shape[0])

    def exact_dft(image, locs):
        """Compute exact Discrete Fourier Transform.

        Parameters
        ----------
        image : torch.Tensor
            Input image with shape (..., Nx, Ny)
        locs : torch.Tensor
            K-space locations with shape (Nk, 2) in [-0.5, 0.5]

        Returns
        -------
        torch.Tensor
            K-space data with shape (..., Nk)
        """
        *batch_dims, Nx, Ny = image.shape
        Nk = locs.shape[0]
        image_flat = image.reshape(*batch_dims, -1)
        x = torch.linspace(-0.5, 0.5, Nx, device=image.device)
        y = torch.linspace(-0.5, 0.5, Ny, device=image.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        r = torch.stack([X.flatten(), Y.flatten()], dim=1)
        r = r.to(image.dtype)
        kspace_data = torch.zeros(
            *batch_dims, Nk, dtype=image.dtype, device=image.device
        )
        chunk_size = 4096
        for _i in range(0, Nk, chunk_size):
            end = min(_i + chunk_size, Nk)
            k_chunk = locs[_i:end]
            phase = -2j * torch.pi * (k_chunk.to(image.dtype) @ r.T)
            fourier_kernel = torch.exp(phase)
            kspace_chunk = image_flat @ fourier_kernel.T
            kspace_data[..., _i:end] = kspace_chunk
        return kspace_data

    return (
        analytic_cartesian_dcf,
        analytic_radial_dcf,
        exact_dft,
        generate_cartesian_trajectory,
        generate_gaussian_coil_sensitivities,
        generate_radial_trajectory,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Generate Ground Truth Image

    We create a simple circular phantom image for testing.
    """)
    return


@app.cell
def _(image_size, plt, torch):
    Nx, Ny = image_size
    # Create a simple circular phantom image
    x_true = torch.zeros(Nx, Ny, dtype=torch.complex64)
    center_x, center_y = (Nx // 2, Ny // 2)
    for _i in range(Nx):
        for j in range(Ny):
            if (_i - center_x) ** 2 + (j - center_y) ** 2 < 33**2:
                x_true[_i, j] = 1.0
    print(f"Ground truth image shape: {x_true.shape}")
    plt.figure(figsize=(6, 6))
    plt.imshow(torch.abs(x_true), cmap="gray", vmin=0, vmax=1)
    plt.title("Ground Truth Phantom (Magnitude)")
    plt.colorbar(label="Signal Intensity")
    plt.axis("off")
    plt.show()
    return (x_true,)


@app.cell
def _(mo):
    mo.md("""
    ## Component 1: Radial Trajectory and NUFFT

    Generate k-space trajectory and create the NUFFT operator for non-Cartesian sampling.
    """)
    return


@app.cell
def _(
    Dim,
    NUFFT,
    generate_cartesian_trajectory,
    generate_radial_trajectory,
    image_size,
    num_readouts,
    num_spokes,
    plt,
    torch,
    use_cartesian,
):
    # Generate k-space trajectory
    if use_cartesian:
        locs = generate_cartesian_trajectory(image_size)
        print(f"Trajectory shape (Cartesian): {locs.shape}")
    else:
        locs = generate_radial_trajectory(num_spokes, num_readouts, image_size)
        print(f"Trajectory shape (Radial): {locs.shape}")

    # Visualize Trajectory
    plt.figure(figsize=(6, 6))
    plt.scatter(locs[:, 0], locs[:, 1], s=0.5, alpha=0.5)
    if use_cartesian:
        plt.title(f"Cartesian K-space Trajectory\n{len(locs)} points")
    else:
        plt.title(
            f"Radial K-space Trajectory\n{num_spokes} spokes, {num_readouts} readouts"
        )
    plt.xlabel("kx (normalized)")
    plt.ylabel("ky (normalized)")
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.grid(True, alpha=0.3)
    plt.show()

    # Scale locs to -N//2, N//2 for NUFFT
    locs = locs * torch.tensor(image_size, dtype=locs.dtype)

    # Create NUFFT operator
    nufft_op = NUFFT(
        locs=locs,
        grid_size=image_size,
        output_shape=Dim("R"),
        input_shape=Dim("NxNy"),
        batch_shape=Dim("C"),
        oversamp=2.0,
        width=6.0,
    )
    return locs, nufft_op


@app.cell
def _(mo):
    mo.md("""
    ## Component 2: Coil Sensitivities (Dense Operator)

    Generate multi-coil sensitivity maps and create the coil encoding operator.
    """)
    return


@app.cell
def _(
    Dense,
    Dim,
    generate_gaussian_coil_sensitivities,
    image_size,
    num_coils,
    plt,
    torch,
):
    coil_sens = generate_gaussian_coil_sensitivities(image_size, num_coils)
    print(f"Coil sensitivities shape: {coil_sens.shape}")
    plt.figure(figsize=(15, 3))
    for _i in range(min(4, num_coils)):
        plt.subplot(1, 4, _i + 1)
        plt.imshow(torch.abs(coil_sens[_i]), cmap="jet")
        plt.title(f"Coil {_i + 1} Sensitivity")
        plt.axis("off")
    plt.suptitle("Coil Sensitivity Maps (Magnitude - First 4 Coils)")
    plt.tight_layout()
    plt.show()

    # Create Dense operator for coil encoding
    coil_op = Dense(
        weight=coil_sens,
        weightshape=Dim("CNxNy"),
        ishape=Dim("NxNy"),
        oshape=Dim("CNxNy"),
    )
    return (coil_op,)


@app.cell
def _(mo):
    mo.md("""
    ## Component 3: Density Compensation (Diagonal Operator)

    Compute density compensation weights for radial trajectories.
    """)
    return


@app.cell
def _(
    Diagonal,
    Dim,
    analytic_cartesian_dcf,
    analytic_radial_dcf,
    locs,
    plt,
    torch,
    use_cartesian,
):
    # Compute analytic density compensation
    if use_cartesian:
        dcf_weights = analytic_cartesian_dcf(locs)
    else:
        dcf_weights = analytic_radial_dcf(locs)

    print(f"Density weights shape: {dcf_weights.shape}")

    # Visualize Density Compensation Weights
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(locs[:, 0], locs[:, 1], c=dcf_weights, s=1, cmap="viridis")
    plt.title("DCF Weights in K-space")
    plt.colorbar(label="Weight")
    plt.axis("equal")

    plt.subplot(1, 2, 2)
    radius = torch.sqrt(locs[:, 0] ** 2 + locs[:, 1] ** 2)
    plt.scatter(radius, dcf_weights, s=1, alpha=0.5)
    plt.title("DCF Weights vs Radius")
    plt.xlabel("Radius (normalized)")
    plt.ylabel("Weight")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Create Diagonal operator for density compensation
    density_op = Diagonal(
        weight=dcf_weights,
        ioshape=Dim("CR"),
    )
    return (density_op,)


@app.cell
def _(mo):
    mo.md("""
    ## Complete Forward Model

    Simulate k-space data from ground truth image using exact DFT.
    """)
    return


@app.cell
def _(coil_op, exact_dft, locs, noise_level, torch, x_true):
    # Simulate k-space data from ground truth image
    multi_coil_image = coil_op(x_true)
    print("Simulating forward model using exact DFT...")
    kspace_data = exact_dft(multi_coil_image, locs)

    # Add realistic noise
    noise = torch.randn_like(kspace_data) * noise_level
    kspace_noisy = kspace_data + noise

    print(f"Forward model: {x_true.shape} -> {kspace_noisy.shape}")
    print(
        f"Data SNR estimate: {20 * torch.log10(torch.norm(kspace_data) / torch.norm(noise)):.1f} dB"
    )
    return (kspace_noisy,)


@app.cell
def _(mo):
    mo.md("""
    ## Iterative Reconstruction

    Solve the least-squares problem using conjugate gradients.
    """)
    return


@app.cell
def _(
    coil_op,
    conjugate_gradients,
    density_op,
    kspace_noisy,
    nufft_op,
    plt,
    power_method,
    sys,
    torch,
    x_true,
):
    # Combine all operators: A = density_comp @ nufft @ coil_sens
    A = (density_op ** (1 / 2)) @ nufft_op @ coil_op

    # Normalize for numerical purposes
    _, eigenval = power_method(
        A.N, torch.ones_like(x_true), tqdm_kwargs=dict(leave=False)
    )
    A = ((1 / (1.01 * eigenval)) ** 0.5) * A

    # Apply density compensation to y
    y = (density_op ** (1 / 2))(kspace_noisy)

    # Compute RHS: A^H y
    rhs = A.H(y)
    rhs = rhs / torch.linalg.vector_norm(rhs)

    plt.figure(figsize=(4, 4))
    plt.imshow(torch.abs(rhs).cpu())
    plt.title("adjoint reconstruction")
    plt.show()

    # Solve A^H A x = A^H y using conjugate gradients
    x_recon = conjugate_gradients(
        A=A.N, y=rhs, max_num_iters=50, gtol=1e-4, tqdm_kwargs=dict(leave=False)
    )

    # Rescale recon to scale of x_true
    if x_recon is not None:
        x_recon = x_recon * torch.max(x_true.abs()) / torch.max(x_recon.abs())
    else:
        print(f"Reconstruction diverged.")
        sys.exit()

    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Ground truth shape: {x_true.shape}")

    # Compute reconstruction error
    recon_error = torch.norm(x_recon - x_true) / torch.norm(x_true)
    print(f"Relative reconstruction error: {recon_error:.4f}")

    # Visualize Reconstruction
    plt.figure(figsize=(15, 5))

    # Ground Truth
    plt.subplot(1, 3, 1)
    plt.imshow(torch.abs(x_true), cmap="gray", vmin=0, vmax=1)
    plt.title("Ground Truth")
    plt.axis("off")

    # Reconstruction
    plt.subplot(1, 3, 2)
    plt.imshow(torch.abs(x_recon), cmap="gray", vmin=0, vmax=1)
    plt.title(f"Reconstruction\n(Error: {recon_error:.4f})")
    plt.axis("off")

    # Error Map
    plt.subplot(1, 3, 3)
    error_map = torch.abs(x_true - x_recon)
    plt.imshow(error_map, cmap="inferno")
    plt.title("Error Map (|x_true - x_recon|)")
    plt.colorbar(label="Error Magnitude")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    return (recon_error,)


@app.cell
def _(mo):
    mo.md("""
    ## Results Summary

    Summary of reconstruction parameters and achieved error.
    """)
    return


@app.cell
def _(
    image_size,
    locs,
    noise_level,
    num_coils,
    num_readouts,
    num_spokes,
    recon_error,
    use_cartesian,
):
    # Print summary statistics
    print("=== MRI Reconstruction Summary ===")
    print(f"Image size: {image_size}")
    print(f"Number of coils: {num_coils}")
    if use_cartesian:
        print(f"Trajectory: Cartesian")
    else:
        print(f"Trajectory: Radial ({num_spokes} spokes × {num_readouts} readouts)")
    print(f"Total k-space samples: {len(locs)}")
    print(f"Undersampling factor: {(image_size[0] * image_size[1]) / len(locs):.2f}×")
    print(f"Noise level: {noise_level}")
    print(f"Relative reconstruction error: {recon_error:.4f}")
    return


if __name__ == "__main__":
    app.run()
