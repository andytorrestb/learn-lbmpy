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

# =====================================================================
# ||                     2) Define domain and initial conditions      ||
# =====================================================================

domain_size = (100, 100)
yHalf = domain_size[1]//2
initial_velocity = np.zeros(domain_size + (2,))
initial_velocity[:, :yHalf, 0] =  0.08
initial_velocity[:, yHalf:, 0] = -0.08
initial_velocity[:, :, 1] += np.random.rand(*domain_size) * 1e-5
plt.figure(dpi=200)
plt.vector_field(initial_velocity, step=8)


# =====================================================================
# ||            2) Configure LBM Models and Run Simulation           ||
# =====================================================================

rr = 1.995
# for the entropic LBM only the shear relaxation rate needs to be specified. Higher order relaxation rates
# are choosen as a symbol and will be subject to an entrpic condition
rr_2 = sp.Symbol("omega_free")

configuration_srt = LBMConfig(method=Method.SRT, relaxation_rate=rr, compressible=True)
periodic_flow_srt = create_fully_periodic_flow(initial_velocity, lbm_config=configuration_srt)

configuration_cumulant = LBMConfig(method=Method.CUMULANT, relaxation_rate=rr, compressible=True)
periodic_flow_cumulant = create_fully_periodic_flow(initial_velocity, lbm_config=configuration_cumulant)

configuration_entropic = LBMConfig(method=Method.MRT, relaxation_rates=[rr, rr, rr_2, rr_2],
                                  compressible=True, entropic=True, zero_centered=False)
periodic_flow_entropic = create_fully_periodic_flow(initial_velocity,
                                                    lbm_config=configuration_entropic)

periodic_flow_entropic.run(1000)
periodic_flow_cumulant.run(1000)
periodic_flow_srt.run(1000)

# =====================================================================
# ||                     3) Visualize results                        ||
# =====================================================================

plt.figure(figsize=(20, 5), dpi=200)
plt.subplot(1, 3, 1)
plt.title("SRT")
plt.scalar_field(periodic_flow_srt.velocity[:, :, 0])
plt.subplot(1, 3, 2)
plt.title("Cumulant")
plt.scalar_field(periodic_flow_cumulant.velocity[:, :, 0])
plt.subplot(1, 3, 3)
plt.title("Entropic")
plt.scalar_field(periodic_flow_entropic.velocity[:, :, 0]);
plt.savefig("fully_periodic_flow_comparison.png")