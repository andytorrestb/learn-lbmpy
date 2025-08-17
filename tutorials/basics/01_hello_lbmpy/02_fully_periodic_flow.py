
# =====================================================================
# ||                     1) Configure enviorment                     ||
# =====================================================================
from pystencils import Target, CreateKernelConfig
from lbmpy.session import *

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

if __name__ == "__main__":
    # =====================================================================
    # ||              2) Set Initial Velocity Field                      ||
    # =====================================================================
    width, height = 200, 60
    velocity_magnitude = 0.05
    init_vel = np.zeros((width,height,2))
    # fluid moving to the right everywhere...
    init_vel[:, :, 0] = velocity_magnitude
    # ...except at a stripe in the middle, where it moves left
    init_vel[:, height//3 : height//3*2, 0] = -velocity_magnitude
    # small random y velocity component
    init_vel[:, :, 1] = 0.1 * velocity_magnitude * np.random.rand(width,height)

    plt.figure(dpi=200)
    plt.vector_field(init_vel, step=4)
    plt.title("Initial Velocity Field")
    plt.savefig("initial_velocity_field.png")
    plt.clf()

    # =====================================================================
    # ||              2) Run simulation and visualize results            ||
    # =====================================================================
    shear_flow_scenario = create_fully_periodic_flow(initial_velocity=init_vel, relaxation_rate=1.97)

    shear_flow_scenario.run(500)
    plt.figure(dpi=200)
    plt.vector_field(shear_flow_scenario.velocity[:, :])
    plt.title("Velocity Field after 500 Iterations")
    plt.savefig("velocity_field_after_500_iterations.png")
    plt.clf()

    plt.figure(dpi=200)
    plt.scalar_field(vorticity_2d(shear_flow_scenario.velocity[:, :]))
    plt.title("Vorticity Field after 500 Iterations")
    plt.savefig("vorticity_field_after_500_iterations.png")
    plt.clf()