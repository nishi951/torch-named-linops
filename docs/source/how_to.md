---
file_format: mystnb
kernelspec:
  name: python3
---
# MRI Reconstruction with torchlinops

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

## Setup and Imports

```{code-cell} python
import torch
from torchlinops import Dense, Diagonal, NUFFT, Dim
from torchlinops.alg import conjugate_gradients

# Set random seed for reproducibility
torch.manual_seed(42)

# Reconstruction parameters
image_size = (128, 128)  # 2D image dimensions
num_coils = 8            # Number of receiver coils
num_spokes = 37          # Number of radial spokes
num_readouts = 256       # Readout points per spoke
noise_level = 0.01       # Additive noise level
```

## Generate Ground Truth Image

```{code-cell} python
# Create a simple phantom image
Nx, Ny = image_size
x_true = torch.zeros(Nx, Ny, dtype=torch.complex64)

# Add some simple structures to the phantom
# Central circle
center_x, center_y = Nx // 2, Ny // 2
for i in range(Nx):
    for j in range(Ny):
        if (i - center_x)**2 + (j - center_y)**2 < 15**2:
            x_true[i, j] = 1.0

# Add some off-center features
x_true[center_x - 20, center_y] = 0.8
x_true[center_x + 20, center_y] = 0.8
x_true[center_x, center_y - 20] = 0.6
x_true[center_x, center_y + 20] = 0.6

print(f"Ground truth image shape: {x_true.shape}")
```

## Component 1: Radial Trajectory and NUFFT

```{code-cell} python
# Generate radial k-space trajectory
locs = generate_radial_trajectory(num_spokes, num_readouts, image_size)
print(f"Trajectory shape: {locs.shape}")  # Should be (num_spokes * num_readouts, 2)

# Create NUFFT operator
nufft_op = NUFFT(
    locs=locs,
    grid_size=image_size,
    output_shape=Dim("R"),  # Readout dimension
    oversamp=1.25,
    width=4.0
)

# Test forward and adjoint operations
test_image = torch.randn(*image_size, dtype=torch.complex64)
kspace_test = nufft_op(test_image)
image_test_adj = nufft_op.H(kspace_test)

print(f"Forward: {test_image.shape} -> {kspace_test.shape}")
print(f"Adjoint: {kspace_test.shape} -> {image_test_adj.shape}")
```

## Component 2: Coil Sensitivities (Dense Operator)

```{code-cell} python
# Generate Gaussian coil sensitivity maps
coil_sens = generate_gaussian_coil_sensitivities(image_size, num_coils)
print(f"Coil sensitivities shape: {coil_sens.shape}")  # Should be (num_coils, Nx, Ny)

# Create Dense operator for coil encoding
# Dense is used because coil sensitivities multiply each coil differently
coil_op = Dense(
    weight=coil_sens,
    weight_shape=Dim("CNxy"),  # Coil × Image dimensions
    ishape=Dim("Nxy"),         # Input image dimensions
    oshape=Dim("CNxy")         # Output multi-coil image dimensions
)

# Test coil encoding
multi_coil_image = coil_op(x_true)
print(f"Coil encoding: {x_true.shape} -> {multi_coil_image.shape}")
```

## Component 3: Density Compensation (Diagonal Operator)

```{code-cell} python
# Compute analytic density compensation for radial trajectories
dcf_weights = analytic_radial_dcf(locs, image_size)
print(f"Density weights shape: {dcf_weights.shape}")

# Create Diagonal operator for density compensation
# Diagonal is used because density compensation is element-wise multiplication
density_op = Diagonal(
    weight=dcf_weights,
    ioshape=Dim("R")  # Same as NUFFT output shape
)

# Test density compensation
kspace_compensated = density_op(kspace_test)
print(f"Density compensation: {kspace_test.shape} -> {kspace_compensated.shape}")
```

## Complete Forward Model

```{code-cell} python
# Combine all operators: A = density_comp @ nufft @ coil_sens
forward_model = density_op @ nufft_op @ coil_op

# Simulate k-space data from ground truth image
kspace_data = forward_model(x_true)

# Add realistic noise
noise = torch.randn_like(kspace_data) * noise_level
kspace_noisy = kspace_data + noise

print(f"Forward model: {x_true.shape} -> {kspace_noisy.shape}")
print(f"Data SNR estimate: {20 * torch.log10(torch.norm(kspace_data) / torch.norm(noise)):.1f} dB")
```

## Iterative Reconstruction

```{code-cell} python
# Create normal operator A^H A for least-squares reconstruction
A_normal = forward_model.N()

# Compute RHS: A^H y
rhs = forward_model.H(kspace_noisy)

# Solve A^H A x = A^H y using conjugate gradients
x_recon = conjugate_gradients(
    A=A_normal,
    y=rhs,
    max_num_iters=50,
    gtol=1e-4
)

print(f"Reconstruction shape: {x_recon.shape}")
print(f"Ground truth shape: {x_true.shape}")

# Compute reconstruction error
recon_error = torch.norm(x_recon - x_true) / torch.norm(x_true)
print(f"Relative reconstruction error: {recon_error:.4f}")
```

## Results Summary

```{code-cell} python
# Print summary statistics
print("=== MRI Reconstruction Summary ===")
print(f"Image size: {image_size}")
print(f"Number of coils: {num_coils}")
print(f"Radial trajectory: {num_spokes} spokes × {num_readouts} readouts")
print(f"Total k-space samples: {len(locs)}")
print(f"Undersampling factor: {(image_size[0] * image_size[1]) / len(locs):.2f}×")
print(f"Noise level: {noise_level}")
print(f"Relative reconstruction error: {recon_error:.4f}")
```

## Appendix: Helper Functions

The following helper functions are used in this tutorial. You can reference them later if you need to understand the implementation details.

```{code-cell} python
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
    # Implementation to be filled later
    pass

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
    # Implementation to be filled later
    pass

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
    # Implementation to be filled later
    pass
```
