from typing import Optional

import torch.nn as nn


class GRAPPAMLP(nn.Module):
    """
    Multi layer perceptron with optional positional encoding.
    Inputs orientation and outputs grappa kernel
    """

    def __init__(self,
                 dim: int,
                 n_coil: int,
                 num_inputs: int,
                 num_layers: int,
                 hidden_width: int,
                 latent_width: Optional[int] = 256,
                 use_b0: Optional[bool] = False,
                 use_batch_norm: Optional[bool] = False,
                 sigma: Optional[float] = 1e-1,
                 n_pos: Optional[int] = 0):
        """
        General GRAPPA MLP.

        Parameters:
        -----------
        dim : int
            dimension of MR data, usually dim = 2 (2D) or dim = 3 (3D)
        n_coil : int
            number of coils
        num_inputs : int
            number of GRAPPA inputs
        num_layers : int
            number of model layers
        hidden_width : int
            width of hidden layers
        latent_width : int
            width of final hidden layer
        use_b0 : bool
            toggles the use of B0 correction, bassically adds a time input
        sigma : float
            standard deviation for positional encoding
        n_pos : int
            number of positional encoding vectors
        """
        super().__init__()

        # Save some consts
        self.dim = dim
        self.n_coil = n_coil
        self.num_inputs = num_inputs
        self.n_pos = n_pos

        # Construct dimensions
        b0_term = 1 if use_b0 else 0
        dimensions = [dim * num_inputs + b0_term] \
                    + [hidden_width] * (num_layers-1) \
                    + [latent_width, n_coil * n_coil * num_inputs]

        # Optional positional encoding
        if n_pos > 0:
            for k in range(1, n_pos):
                B = torch.tensor(np.random.normal(0, sigma, (k, n_pos))).float()
                self.register_buffer(f'B_{k}', B)
            dimensions[0] = n_pos * 2

        # Define layers from inputs
        self.feature_layers = nn.ModuleList()
        for k in range(0, len(dimensions) - 2):

            # Mat mul
            self.feature_layers.append(nn.Linear(dimensions[k], dimensions[k+1], dtype=torch.float32))

            # Batch norm on all relus
            if use_batch_norm:
                self.feature_layers.append(nn.BatchNorm1d(dimensions[k+1]))

            # Activation
            self.feature_layers.append(nn.ReLU())

        # Last layer
        self.last_layer = nn.Linear(dimensions[-2], dimensions[-1], dtype=torch.complex64)

    def forward(self, orientations, source_pts):
        """
        Forward pass of model

        Parameters:
        -----------
        orientations : torch.tensor <float>
            GRAPPA orientation vectors with shape (N, num_inputs * dim) N is batch
        source_pts : torch.tensor <complex>
            GRAPPA source points with shape (N, num_inputs * n_coil)

        Returns:
        ----------
        target : torch.tensor <complex>
            output target points with shape (N, n_coil)
        grappa_kern : torch.tensor <complex>
            grappa kernel with shape (N, n_coil, n_coil * num_inputs)
        """

        # Positional encoding
        if self.n_pos > 0:
            num_inputs = orientations.shape[1]
            pos_enc_cos = torch.cos(2 * np.pi * orientations @ eval(f'self.B_{num_inputs}'))
            pos_enc_sin = torch.sin(2 * np.pi * orientations @ eval(f'self.B_{num_inputs}'))
            orientations = torch.hstack((pos_enc_sin, pos_enc_cos))

        # Feature extraction
        for layer in self.feature_layers:
            orientations = layer(orientations)

        # Linear comb kernel
        kern_flat = self.last_layer(orientations.type(torch.complex64))

        # Reshape to grappa kernel
        grappa_kern = rearrange(kern_flat, '... (nc nc_ninp) -> ... nc nc_ninp',
                                nc=self.n_coil,
                                nc_ninp=self.n_coil * self.num_inputs)

        # Apply to data
        target = (grappa_kern @ source_pts[..., None])[..., 0]

        return target, grappa_kern
