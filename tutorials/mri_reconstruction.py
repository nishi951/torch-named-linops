"""
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

$$\min_x \|A x - y\|_2^2$$

where $A = E F S$ is the combined forward operator.
"""

# %%
# Setup and Imports
# -----------------
import sys
import warnings

import matplotlib.pyplot as plt
import torch

from torchlinops import NUFFT, Dense, Diagonal, Dim, Identity
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

# %%
# Helper Functions
# ----------------
# We define the helper functions here for generating trajectories, sensitivities, and density compensation weights.


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

    # Generate angles for each spoke (evenly distributed around circle)
    # Note: endpoint=False not available in older PyTorch, so we exclude last point manually
    angles = torch.linspace(0, 2 * torch.pi, num_spokes + 1)[:-1]

    # Generate readout positions (from -max_radius to +max_radius)
    readout_pos = torch.linspace(-max_radius, max_radius, num_readouts)

    # Initialize trajectory
    locs = torch.zeros(num_spokes * num_readouts, 2)

    # Fill trajectory
    for i, angle in enumerate(angles):
        start_idx = i * num_readouts
        end_idx = (i + 1) * num_readouts

        # Convert to Cartesian coordinates
        x = readout_pos * torch.cos(angle)
        y = readout_pos * torch.sin(angle)

        locs[start_idx:end_idx, 0] = x
        locs[start_idx:end_idx, 1] = y

    # Normalize to [-0.5, 0.5] range (standard for NUFFT)
    locs = locs / (2 * max_radius)

    return locs


def generate_cartesian_trajectory(grid_size, oversamp=2.0, resolution_scale=2.0):
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

    # Create coordinate grids
    x = torch.linspace(-0.5, 0.5, Nx)
    y = torch.linspace(-0.5, 0.5, Ny)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Initialize coil sensitivities
    coil_sens = torch.zeros(num_coils, Nx, Ny, dtype=torch.complex64)

    # Generate Gaussian sensitivities with different centers
    for c in range(num_coils):
        # Place coil centers around the image in a circular pattern
        angle = 2 * torch.pi * c / num_coils
        center_x = 0.3 * torch.cos(torch.tensor(angle))
        center_y = 0.3 * torch.sin(torch.tensor(angle))

        # Create Gaussian sensitivity profile
        sigma = 1.0  # Width of Gaussian
        gaussian = torch.exp(
            -((X - center_x) ** 2 + (Y - center_y) ** 2) / (2 * sigma**2)
        )

        # Add phase variation for more realistic coil sensitivities
        phase = torch.exp(1j * 2 * torch.pi * (0.1 * X + 0.1 * Y))

        coil_sens[c] = gaussian * phase

    return coil_sens


def analytic_radial_dcf(locs, grid_size):
    """Compute analytic density compensation for radial trajectories.

    Parameters
    ----------
    locs : torch.Tensor
        K-space locations with shape (N, 2)
    grid_size : tuple
        Image grid size (Nx, Ny)

    Returns
    -------
    torch.Tensor
        Density compensation weights with shape (N,)
    """
    Nx, Ny = grid_size

    # Compute radius for each k-space location
    radius = torch.sqrt(locs[:, 0] ** 2 + locs[:, 1] ** 2)

    # For radial trajectories, density is proportional to radius
    # This is because outer regions have fewer samples per unit area
    # The analytic DCF for radial is: dcf = radius
    # We add a small constant to avoid division by zero at center
    dcf = radius + 1e-6

    # Normalize to have unit mean
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

    # Flatten image spatial dimensions
    image_flat = image.reshape(*batch_dims, -1)  # (..., Nx*Ny)

    # Generate spatial coordinates
    # Note: Use same convention as NUFFT (centered grid)
    x = torch.linspace(-0.5, 0.5, Nx, device=image.device)
    y = torch.linspace(-0.5, 0.5, Ny, device=image.device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    r = torch.stack([X.flatten(), Y.flatten()], dim=1)  # (Nx*Ny, 2)
    r = r.to(image.dtype)  # Ensure type match

    # Compute DFT in chunks to save memory
    kspace_data = torch.zeros(*batch_dims, Nk, dtype=image.dtype, device=image.device)
    chunk_size = 4096

    for i in range(0, Nk, chunk_size):
        end = min(i + chunk_size, Nk)
        k_chunk = locs[i:end]  # (Chunk, 2)

        # Compute phase: -2*pi*i * (k . r)
        # k: (Chunk, 2), r: (Pixels, 2) -> (Chunk, Pixels)
        # Cast to complex64/128 for phase calculation
        phase = -2j * torch.pi * (k_chunk.to(image.dtype) @ r.T)
        fourier_kernel = torch.exp(phase)  # (Chunk, Pixels)

        # Apply kernel: sum over pixels
        # image_flat: (..., Pixels), kernel: (Chunk, Pixels)
        # result: (..., Chunk)
        kspace_chunk = image_flat @ fourier_kernel.T

        kspace_data[..., i:end] = kspace_chunk

    return kspace_data


# %%
# Generate Ground Truth Image
# ---------------------------

# Create a simple circular phantom image
Nx, Ny = image_size
x_true = torch.zeros(Nx, Ny, dtype=torch.complex64)
center_x, center_y = Nx // 2, Ny // 2
for i in range(Nx):
    for j in range(Ny):
        if (i - center_x) ** 2 + (j - center_y) ** 2 < 33**2:
            x_true[i, j] = 1.0


print(f"Ground truth image shape: {x_true.shape}")

# Visualize Ground Truth
plt.figure(figsize=(6, 6))
plt.imshow(torch.abs(x_true), cmap="gray", vmin=0, vmax=1)
plt.title("Ground Truth Phantom (Magnitude)")
plt.colorbar(label="Signal Intensity")
plt.axis("off")
plt.show()


# %%
# Component 1: Radial Trajectory and NUFFT
# ----------------------------------------

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
    output_shape=Dim("R"),  # Readout dimension
    input_shape=Dim("CNxNy"),
    oversamp=2.0,
    width=6.0,
)

# Test forward and adjoint operations
# test_image = torch.randn(*image_size, dtype=torch.complex64)
# kspace_test = nufft_op(test_image)
# image_test_adj = nufft_op.H(kspace_test)

# print(f"Forward: {test_image.shape} -> {kspace_test.shape}")
# print(f"Adjoint: {kspace_test.shape} -> {image_test_adj.shape}")


# %%
# Component 2: Coil Sensitivities (Diagonal Operator)
# ---------------------------------------------------

# Generate Gaussian coil sensitivity maps
coil_sens = generate_gaussian_coil_sensitivities(image_size, num_coils)
print(f"Coil sensitivities shape: {coil_sens.shape}")  # Should be (num_coils, Nx, Ny)

# Visualize Coil Sensitivities
plt.figure(figsize=(15, 3))
for i in range(min(4, num_coils)):
    plt.subplot(1, 4, i + 1)
    plt.imshow(torch.abs(coil_sens[i]), cmap="jet")
    plt.title(f"Coil {i + 1} Sensitivity")
    plt.axis("off")
plt.suptitle("Coil Sensitivity Maps (Magnitude - First 4 Coils)")
plt.tight_layout()
plt.show()

# Create Dense operator for coil encoding
coil_op = Dense(
    weight=coil_sens,
    weightshape=Dim("CNxNy"),  # Coil × Image dimensions
    ishape=Dim("NxNy"),  # Input image dimensions
    oshape=Dim("CNxNy"),  # Output multi-coil image dimensions
)

# Test coil encoding
# multi_coil_image = coil_op(x_true)
# print(f"Coil encoding: {x_true.shape} -> {multi_coil_image.shape}")


# %%
# Component 3: Density Compensation (Diagonal Operator)
# -----------------------------------------------------

# Compute analytic density compensation
if use_cartesian:
    dcf_weights = analytic_cartesian_dcf(locs)
else:
    dcf_weights = analytic_radial_dcf(locs, image_size)

print(f"Density weights shape: {dcf_weights.shape}")

# Visualize Density Compensation Weights
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(locs[:, 0], locs[:, 1], c=dcf_weights, s=1, cmap="viridis")
plt.title("DCF Weights in K-space")
plt.colorbar(label="Weight")
plt.axis("equal")

plt.subplot(1, 2, 2)
# Plot weights vs radius to verify radial dependency
radius = torch.sqrt(locs[:, 0] ** 2 + locs[:, 1] ** 2)
plt.scatter(radius, dcf_weights, s=1, alpha=0.5)
plt.title("DCF Weights vs Radius")
plt.xlabel("Radius (normalized)")
plt.ylabel("Weight")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Create Diagonal operator for density compensation
# Diagonal is used because density compensation is element-wise multiplication
density_op = Diagonal(
    weight=dcf_weights,
    ioshape=Dim("R"),  # Same as NUFFT output shape
)

# Test density compensation
# kspace_compensated = density_op(kspace_test)
# print(f"Density compensation: {kspace_test.shape} -> {kspace_compensated.shape}")


# %%
# Complete Forward Model
# ----------------------

# Simulate k-space data from ground truth image
# We use exact DFT simulation to avoid inverse crimes
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


# %%
# Iterative Reconstruction
# ------------------------

# Combine all operators: A = density_comp @ nufft @ coil_sens
A = (density_op ** (1 / 2)) @ nufft_op @ coil_op

# Normalize for numerical purposes
_, eigenval = power_method(A.N, torch.ones_like(x_true), tqdm_kwargs=dict(leave=False))
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
# Note: A_normal is a NamedLinop which is callable, so we can pass it directly
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

# %%
# Results Summary
# ---------------

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
