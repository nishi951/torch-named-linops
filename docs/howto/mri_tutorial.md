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

## Setup and Imports

```python
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

```python
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

```python
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

## Component 2: Coil Sensitivities (Diagonal Operator)

```python
# Generate Gaussian coil sensitivity maps
coil_sens = generate_gaussian_coil_sensitivities(image_size, num_coils)
print(f"Coil sensitivities shape: {coil_sens.shape}")  # Should be (num_coils, Nx, Ny)

# Create Dense operator for coil encoding
coil_op = Dense(
    weight=coil_sens,
    weight_shape=Dim("CNxNy"),  # Coil × Image dimensions
    ishape=Dim("NxNy"),         # Input image dimensions
    oshape=Dim("CNxNy")         # Output multi-coil image dimensions
)

# Test coil encoding
# Dense operator needs proper input handling for multi-coil encoding
# We need to expand the single-coil image for Dense to work properly
x_expanded = x_true.unsqueeze(0).expand(num_coils, -1, -1)
multi_coil_image = coil_op(x_expanded)
print(f"Coil encoding: {x_expanded.shape} -> {multi_coil_image.shape}")
```

## Component 3: Density Compensation (Diagonal Operator)

```python
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

```python
# Combine all operators: A = density_comp @ nufft @ coil_sens
forward_model = density_op @ nufft_op @ coil_op

# Simulate k-space data from ground truth image
# Need to expand single-coil image to multi-coil for coil encoding
x_expanded = x_true.unsqueeze(0).expand(num_coils, -1, -1)
kspace_data = forward_model(x_expanded)

# Add realistic noise
noise = torch.randn_like(kspace_data) * noise_level
kspace_noisy = kspace_data + noise

print(f"Forward model: {x_true.shape} -> {kspace_noisy.shape}")
print(f"Data SNR estimate: {20 * torch.log10(torch.norm(kspace_data) / torch.norm(noise)):.1f} dB")
```

## Iterative Reconstruction

```python
# Create normal operator A^H A for least-squares reconstruction
A_normal = forward_model.N  # Access property, not call method

# Compute RHS: A^H y
rhs = forward_model.H(kspace_noisy)

# Solve A^H A x = A^H y using conjugate gradients
# Note: A_normal is a NamedLinop which is callable, so we can pass it directly
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

```python
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

```python
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
        sigma = 0.25  # Width of Gaussian
        gaussian = torch.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        
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
    radius = torch.sqrt(locs[:, 0]**2 + locs[:, 1]**2)
    
    # For radial trajectories, density is proportional to radius
    # This is because outer regions have fewer samples per unit area
    # The analytic DCF for radial is: dcf = radius
    # We add a small constant to avoid division by zero at center
    dcf = radius + 1e-6
    
    # Normalize to have unit mean
    dcf = dcf / torch.mean(dcf)
    
    return dcf
```
