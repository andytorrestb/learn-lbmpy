from pystencils import Target
from lbmpy.session import *


# =====================================================================
# ||                     1) Helper Functions                         ||
# =====================================================================

def pipe_geometry_callback(x, y, z):
    radius = domain_size[1] / 2
    y_mid = domain_size[1] / 2
    z_mid = domain_size[2] / 2
    return (y - y_mid) ** 2 + (z - z_mid) ** 2 > radius ** 2

def velocity_info_callback(boundary_data, activate=True, **_):
    boundary_data['vel_1'] = 0
    boundary_data['vel_2'] = 0

    if activate:
        u_max = 0.1
        y, z = boundary_data.link_positions(1), boundary_data.link_positions(2)
        radius = domain_size[1] // 2
        centered_y = y - radius
        centered_z = z - radius
        dist_to_center = np.sqrt(centered_y**2 + centered_z**2)
        boundary_data['vel_0'] = u_max * (1 - dist_to_center / radius)
    else:
        boundary_data['vel_0'] = 0

if __name__ == "__main__":

    # =====================================================================
    # ||                     2) Create 3D LBM Domain                     ||
    # =====================================================================

    domain_size = (64, 16, 16)
    lbm_config = LBMConfig(stencil=Stencil.D3Q27, method=Method.SRT, relaxation_rate=1.9)
    config = CreateKernelConfig(target=Target.CPU)
    # config = CreateKernelConfig(target=Target.GPU, gpu_indexing_params={'block_size': (128, 1, 1)})

    sc2 = LatticeBoltzmannStep(domain_size=domain_size,
                           lbm_config=lbm_config,
                           config=config)
    
    # =====================================================================
    # ||                 3) Configure Boundary Conditions                ||
    # =====================================================================

    inflow = UBB(velocity_info_callback, dim=sc2.method.dim)

    stencil = LBStencil(Stencil.D3Q27)
    outflow = ExtrapolationOutflow(stencil[4], sc2.method)

    sc2.boundary_handling.set_boundary(inflow, make_slice[0, :, :])
    sc2.boundary_handling.set_boundary(outflow, make_slice[-1, :, :])

    wall = NoSlip()
    sc2.boundary_handling.set_boundary(wall, mask_callback=pipe_geometry_callback)

    # =====================================================================
    # ||                 4) Plot Boundary Conditions                     ||
    # =====================================================================

    plt.rc('figure', figsize=(14, 8), dpi=200)

    plt.subplot(231)
    plt.boundary_handling(sc2.boundary_handling, make_slice[0, :, :], show_legend=False)
    plt.title('Inflow')

    plt.subplot(232)
    plt.boundary_handling(sc2.boundary_handling, make_slice[0.5, :, :], show_legend=False)
    plt.title('Middle')

    plt.subplot(233)
    plt.boundary_handling(sc2.boundary_handling, make_slice[-1, :, :], show_legend=False)
    plt.title('Outflow')

    plt.subplot(212)
    plt.boundary_handling(sc2.boundary_handling, make_slice[:, 0.5, :], show_legend=True)
    plt.title('Cross section')
    plt.savefig('boundary_conditions.png')
    plt.clf()

    # =====================================================================
    # ||                 5) Run For a Few Steps                          ||
    # =====================================================================
    n_steps = 20
    sc2.run(n_steps)
    plt.figure(dpi=200)
    plt.scalar_field(sc2.velocity[:, 0.5, :, 0])
    plt.colorbar()
    plt.savefig(f'velocity_field_after_{n_steps}_steps.png')
    plt.clf()

    # =====================================================================
    # ||        6) Turn off Inflow Condition and Keep Running             ||
    # =====================================================================
    sc2.boundary_handling.trigger_reinitialization_of_boundary_data(activate=False)
    n_steps = 50
    sc2.run(n_steps)
    plt.figure(dpi=200)
    plt.scalar_field(sc2.velocity[:, 0.5, :, 0])
    plt.colorbar()
    plt.savefig(f'velocity_field_after_{n_steps}_steps.png')
    plt.clf()
