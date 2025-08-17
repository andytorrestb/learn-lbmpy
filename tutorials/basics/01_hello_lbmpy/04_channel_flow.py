
# =====================================================================
# ||                     1) Configure enviorment                     ||
# =====================================================================
from pystencils import Target, CreateKernelConfig
from lbmpy.session import *

from lbmpy.boundaries import NoSlip

try:
    import cupy
except ImportError:
    cupy = None
    gpu = False
    target = ps.Target.CPU
    print('No cupy installed')

if cupy:
    gpu = True
    target = ps.Target.GPU

# =====================================================================
# ||                     2) Helper Functions                         ||
# =====================================================================

def draw_boundary_setup(case, case_name):
    fig = plt.figure(figsize=(10.0, 3.0), dpi=200)
    plt.boundary_handling(case.boundary_handling)
    plt.axis('off')
    plt.savefig(f"channel_boundary_setup_{case_name}.png")
    plt.clf()

def set_sphere(x, y):
    shape = channel_scenario.domain_size
    mid = (0.5 * shape[0], 0.5 * shape[1])
    radius = 13
    return (x-mid[0])**2 + (y-mid[1])**2 < radius**2

if __name__ == "__main__":

    # =====================================================================
    # ||                3) Create and Run Channel Flow                   ||
    # =====================================================================

    channel_scenario = create_channel(domain_size=(300, 100), force=1e-7, initial_velocity=(0.025, 0),
                                  relaxation_rate=1.97,
                                  config=CreateKernelConfig(target=Target.CPU))
    print(channel_scenario._lbmKernels[0].ast)

    channel_scenario.run(10000)
    plt.figure(dpi=200)
    plt.vector_field(channel_scenario.velocity[:, :], step=4)
    plt.savefig("channel_flow.png")
    plt.clf()

    vel_profile = channel_scenario.velocity[0.5, :, 0]
    plt.figure(dpi=200)
    plt.plot(vel_profile)
    plt.savefig("channel_flow_profile.png")
    plt.clf()

    draw_boundary_setup(channel_scenario, 'base')

    # =====================================================================
    # ||                4) Add No Slip Obstacle                          ||
    # =====================================================================

    wall = NoSlip()
    channel_scenario.boundary_handling.set_boundary(wall, make_slice[0.2:0.25, 0:0.333])

    draw_boundary_setup(channel_scenario, 'wall')

    # =====================================================================
    # ||                5) Add Cut-Out to No Slip Obstacle               ||
    # =====================================================================

    channel_scenario.boundary_handling.set_boundary('domain', make_slice[0.2:0.235, 0.0333:0.3])
    fig = plt.figure(figsize=(10.0, 3.0), dpi=200)
    plt.boundary_handling(channel_scenario.boundary_handling)
    plt.axis('off')
    plt.savefig(f"channel_boundary_setup_cut_out.png")
    plt.clf()

    # =====================================================================
    # ||                5) Add Sphere as a No Slip Obstacle              ||
    # =====================================================================

    channel_scenario.boundary_handling.set_boundary(wall, mask_callback=set_sphere)
    draw_boundary_setup(channel_scenario, 'sphere')

    channel_scenario.run(10000)
    plt.figure(dpi=200)
    plt.vector_field(channel_scenario.velocity[:, :], step=4)
    plt.savefig("channel_flow_with_obstacle.png")
    plt.clf()