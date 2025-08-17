from lbmpy.session import *
from lbmpy.relaxationrates import relaxation_rate_from_lattice_viscosity
from lbmpy.macroscopic_value_kernels import pdf_initialization_assignments

def set_sphere(x, y, *_):
    mid = (domain_size[0] // 3, domain_size[1] // 2)
    radius = reference_length // 2
    return (x-mid[0])**2 + (y-mid[1])**2 < radius**2

def timeloop(timeSteps):
    for i in range(timeSteps):
        bh()
        dh.run_kernel(kernel)
        dh.swap("src", "dst")

if __name__ == "__main__":
    # # Part A) 
    # lbm_config = LBMConfig(method=Method.MONOMIAL_CUMULANT, relaxation_rate=sp.Symbol('omega_v'), compressible=True)
    # method_monomial = create_lb_method(lbm_config=lbm_config)
    # print(method_monomial)  

    # lbm_config = LBMConfig(method=Method.CUMULANT, relaxation_rate=sp.Symbol('omega_v'), compressible=True)
    # method_polynomial = create_lb_method(lbm_config=lbm_config)
    # print(method_polynomial)

    # Step 1) Define relation rate based on physical parameters.
    reference_length = 30
    maximal_velocity = 0.05
    reynolds_number = 100000
    kinematic_vicosity = (reference_length * maximal_velocity) / reynolds_number
    initial_velocity=(maximal_velocity, 0)

    omega = relaxation_rate_from_lattice_viscosity(kinematic_vicosity)

    # Step 2) Define the stencil and domain size.
    stencil = LBStencil(Stencil.D2Q9)
    domain_size = (reference_length * 12, reference_length * 4)
    dim = len(domain_size)

    # Step 3)  Allocate data arrays  for flow field data.
    dh = ps.create_data_handling(domain_size=domain_size, periodicity=(False, False))

    # These arrays are needed to implement the two grid pull pattern.
    src = dh.add_array('src', values_per_cell=len(stencil), alignment=True)
    dh.fill('src', 0.0, ghost_layers=True)
    dst = dh.add_array('dst', values_per_cell=len(stencil), alignment=True)
    dh.fill('dst', 0.0, ghost_layers=True)

    velField = dh.add_array('velField', values_per_cell=dh.dim, alignment=True)
    dh.fill('velField', 0.0, ghost_layers=True)

    # Step 4) Configure LBM Model
    lbm_config = LBMConfig(stencil=Stencil.D2Q9, method=Method.CUMULANT, relaxation_rate=omega,
                       compressible=True,
                       output={'velocity': velField}, kernel_type='stream_pull_collide')

    method = create_lb_method(lbm_config=lbm_config)
    print(method)

    # Step 5) Initialize PDF with equilibrium distribution functions
    init = pdf_initialization_assignments(method, 1.0, initial_velocity, src.center_vector)

    ast_init = ps.create_kernel(init, target=dh.default_target)
    kernel_init = ast_init.compile()
    dh.run_kernel(kernel_init)

    # Step 6) Define the Update Rule
    lbm_optimisation = LBMOptimisation(symbolic_field=src, symbolic_temporary_field=dst)
    update = create_lb_update_rule(lb_method=method,
                                lbm_config=lbm_config,
                                lbm_optimisation=lbm_optimisation)

    ast_kernel = ps.create_kernel(update, target=dh.default_target, cpu_openmp=True)
    kernel = ast_kernel.compile()

    # Step 7) Set Up and Plot Boundary Conditions
    bh = LatticeBoltzmannBoundaryHandling(method, dh, 'src', name="bh")

    inflow = UBB(initial_velocity)
    outflow = ExtrapolationOutflow(stencil[4], method)
    wall = NoSlip("wall")

    bh.set_boundary(inflow, slice_from_direction('W', dim))
    bh.set_boundary(outflow, slice_from_direction('E', dim))
    for direction in ('N', 'S'):
        bh.set_boundary(wall, slice_from_direction(direction, dim))

    bh.set_boundary(NoSlip("obstacle"), mask_callback=set_sphere)

    plt.figure(dpi=200)
    plt.boundary_handling(bh)
    plt.savefig("boundary_conditions.png")
    plt.clf()

    # Step 8): Run the Simulation
    mask = np.fromfunction(set_sphere, (domain_size[0], domain_size[1], len(domain_size)))
    if 'is_test_run' not in globals():
        timeloop(50000)  # initial steps

        def run():
            timeloop(100)
            return np.ma.array(dh.gather_array('velField'), mask=mask)

        animation = plt.vector_field_magnitude_animation(run, frames=600, rescale=True)
        
        # Save animation to MP4 file immediately
        print("Saving animation to MP4...")
        try:
            animation.save("cumulant_lbm_animation.mp4", writer='ffmpeg', fps=30, bitrate=1800)
            print("Animation saved as 'cumulant_lbm_animation.mp4'")
        except Exception as e:
            print(f"Failed to save MP4: {e}")
            # Try alternative approach
            try:
                animation.save("cumulant_lbm_animation.mp4", writer='pillow', fps=30)
                print("Animation saved as MP4 using pillow writer")
            except Exception as e2:
                print(f"Both MP4 methods failed: {e2}")
                # Fallback to GIF
                try:
                    animation.save("cumulant_lbm_animation.gif", writer='pillow', fps=30)
                    print("Animation saved as GIF instead")
                except Exception as e3:
                    print(f"All save methods failed: {e3}")
        
        type(animation)
        input()
        # set_display_mode('video')
        # res = display_animation(animation)
    else:
        timeloop(10)
        res = None



