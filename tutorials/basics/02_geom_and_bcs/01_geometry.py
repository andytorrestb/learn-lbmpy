from pystencils import Target
from lbmpy.session import *

def pipe_geometry_callback(x, y, z):
    radius = domain_size[1] / 2
    y_mid = domain_size[1] / 2
    z_mid = domain_size[2] / 2
    return (y - y_mid) ** 2 + (z - z_mid) ** 2 > radius ** 2


if __name__ == "__main__":
    domain_size = (64, 16, 16)

    lbm_config = LBMConfig(stencil=Stencil.D3Q27, method=Method.SRT, relaxation_rate=1.9, force=(1e-6, 0, 0))
    config = CreateKernelConfig(target=Target.CPU)

    sc1 = LatticeBoltzmannStep(domain_size=domain_size,
                            periodicity=(True, False, False),
                            lbm_config=lbm_config,
                            config=config)

    wall = NoSlip()
sc1.boundary_handling.set_boundary(wall, mask_callback=pipe_geometry_callback)

plt.figure(dpi=200)
plt.boundary_handling(sc1.boundary_handling, make_slice[0.5, :, :])
plt.savefig("pipe_boundary_setup.png")
plt.clf()

sc1.run(500)
plt.figure(dpi=200)
plt.scalar_field(sc1.velocity[domain_size[0] // 2, :, :, 0])
plt.savefig("pipe_velocity_profile.png")
plt.clf()

    