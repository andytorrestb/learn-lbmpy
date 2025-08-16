"""
Simple Lid-Driven Cavity Flow - LBMPY Hello World

This example shows the basics of running an LBM simulation:
1. Create a simulation using a preset configuration
2. Run it for some steps
3. Visualize the results

Source: https://pycodegen.pages.i10git.cs.fau.de/lbmpy/notebooks/00_tutorial_lbmpy_walberla_overview.html

"""


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

# Create simulation (explain what these parameters mean)
simulation =  create_lid_driven_cavity(domain_size=(100, 100),
                               lbm_config=LBMConfig(method=Method.SRT, relaxation_rate=1.6))


# =====================================================================
# ||                     2) Run simulation                           ||
# =====================================================================
print("Running simulation...")
simulation.run(100)  # 100 time steps


# =====================================================================
# ||                     3) Visualize results                        ||
# =====================================================================
plt.figure(figsize=(8, 6))
plt.vector_field(simulation.velocity_slice(), step=3)
plt.title("Velocity Field in Lid-Driven Cavity")
plt.savefig("lid_driven_cavity_srt.png")
plt.clf()

# =====================================================================
# ||                     4) Repeat process using MRT               ||
# =====================================================================

# Create simulation (explain what these parameters mean)
simulation =  create_lid_driven_cavity(domain_size=(100, 100),
                               lbm_config=LBMConfig(method=Method.MRT, relaxation_rate=1.6))
print("Running simulation...")
simulation.run(100)  # 100 time steps

plt.figure(figsize=(8, 6))
plt.vector_field(simulation.velocity_slice(), step=3)
plt.title("Velocity Field in Lid-Driven Cavity")
plt.savefig("lid_driven_cavity_mrt.png")
plt.clf()