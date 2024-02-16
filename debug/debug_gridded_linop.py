import sigpy as sp
import sigpy.mri as mri

import torchlinops.mri.app as app

# Simulate data
data = app.Simulate.from_params().run()

# Extract calibration
data.mps_recon, data.kgrid_recon = \
    app.Calib(
        data.trj,
        data.ksp,
        data.mps.shape[1:],
        device=sp.Device(0),
    ).run()

# Run Naive Recon
cgsense_recon = app.CGSENSE(
    data.ksp,
    data.trj,
    data.mps_recon,
    # CGSENSE params
).run()

# Run FISTA+LLR Recon
prior = LocallyLowRank()
fistallr_recon = app.FISTA(
    data.ksp,
    data.trj,
    data.mps_recon,
    prior=prior,
    # FISTA Params
).run()

###

# Grid trj and ksp
data.ksp_grid, data.trj_grid = app.VanillaImplicitGROG(
    data.kgrid_recon,   # Calibration region
    data.ksp,           # Kspace data
    data.trj,           # Trajectory
).run()

# Rerun the old recons
cgsense_grid_recon = app.CGSENSE(
    data.ksp_grid,
    data.trj_grid,
    data.mps_recon,
).run()

fistallr_grid_recon = app.FISTA(
    data.ksp_grid,
    data.trj_grid,
    data.mps_recon,
    prior=prior,
    # FISTA params
).run()



breakpoint()
