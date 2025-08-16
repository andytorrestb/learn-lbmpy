
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

    # ============= 1) Create Lid-Driven Cavity Scenario ==============
    ldc_scenario = create_lid_driven_cavity(domain_size=(80,50), lid_velocity=0.01, relaxation_rate=1.95)
    ldc_scenario.method

    ldc_scenario.run(2000)
    plt.figure(dpi=200)
    plt.vector_field(ldc_scenario.velocity_slice(), step=2)
    plt.title("Velocity Field in Lid-Driven Cavity")
    plt.savefig("lid_driven_cavity_srt.png")
    plt.clf()

    # ============= 2) Re-run for varying relaxation rates ==============
    # NOTE: This section is a WIP. It will analysize  the impact of relaxation rate (i.e. Reynolds number). Completion will require comparison
    #       to experimental results and assessment of steady state conditions.
    for relaxation_rate in [1.96, 1.97, 1.98, 1.99, 2.00]:
        ldc_scenario = create_lid_driven_cavity(domain_size=(80,50), lid_velocity=0.01, relaxation_rate=relaxation_rate)
        ldc_scenario.run(2000)
        plt.figure(dpi=200)
        plt.vector_field(ldc_scenario.velocity_slice(), step=2)
        plt.title(f"Velocity Field in Lid-Driven Cavity (Relaxation Rate: {relaxation_rate})")
        plt.savefig(f"lid_driven_cavity_relaxation_{relaxation_rate}.png")
        plt.clf()

    # ============= 3) 3D Lid-Driven Cavity ==============
    # NOTE: This section is a WIP. It will expand the simulation to model 3D flow.
    # We need to decide how we can expand on the base examples to explore lbmpy.
    ldc_scenario = create_lid_driven_cavity(domain_size=(80,50,30), lid_velocity=0.01, relaxation_rate=1.95)
    ldc_scenario.run(2000)
    plt.figure(dpi=200)
    plt.vector_field(ldc_scenario.velocity[:, :, 10, 0:2], step=2)
    plt.title("Velocity Field in 3D Lid-Driven Cavity")
    plt.savefig("lid_driven_cavity_3d.png")
    plt.clf()