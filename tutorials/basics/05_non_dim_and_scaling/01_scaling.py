from lbmpy.session import *
from lbmpy.parameterization import ScalingWidget
from lbmpy.parameterization import Scaling

if __name__ == "__main__":
    p = ScalingWidget()

    cm = 1 / 100
    sc = Scaling(physical_length=0.5 * cm, physical_velocity=2*cm, kinematic_viscosity=1e-6,
             cells_per_length=30)
    scaling_result = sc.diffusive_scaling(1.9)
    print(scaling_result)

    total_time_steps = round(3 / scaling_result.dt)
    print(f'total_time_steps: {total_time_steps}')

    print(sc.acoustic_scaling(dt=1e-4))
    print(sc.fixed_lattice_velocity_scaling(0.1))

    print(sc.reynolds_number, sc.dx)

    domain_size_in_cells = (round(6*cm / sc.dx), round(2*cm / sc.dx))
    domain_size_in_cells

    scenario1 = create_channel(domain_size_in_cells, u_max=scaling_result.lattice_velocity,
                            relaxation_rate=1.9, optimization={'openmp': 4})

    obstacle_midpoint = (round(2 * cm / sc.dx),
                        round(0.8*cm / sc.dx))
    obstacle_radius = round(0.5 * cm / 2 / sc.dx)
    add_sphere(scenario1.boundary_handling, obstacle_midpoint, obstacle_radius, NoSlip("obstacle"))
    plt.figure(dpi=200)
    plt.boundary_handling(scenario1.boundary_handling)
    plt.savefig("boundary_conditions.png")
    plt.clf()

    print(domain_size_in_cells)
    print(obstacle_radius)
    print(obstacle_midpoint)

    # Create obstacle mask for velocity field visualization
    def set_obstacle_mask(x, y, *_):
        return (x-obstacle_midpoint[0])**2 + (y-obstacle_midpoint[1])**2 < obstacle_radius**2

    if 'is_test_run' not in globals():
        scenario1.run(30000)  # initial steps

        def run():
            scenario1.run(100)
            return scenario1.velocity[:, :]

        animation = plt.vector_field_magnitude_animation(run, frames=600, rescale=True)
        
        # Save animation to MP4 file immediately
        print("Saving animation to MP4...")
        try:
            animation.save("scaling_simulation_animation.mp4", writer='ffmpeg', fps=30, bitrate=1800)
            print("Animation saved as 'scaling_simulation_animation.mp4'")
        except Exception as e:
            print(f"Failed to save MP4: {e}")
            # Try alternative approach
            try:
                animation.save("scaling_simulation_animation.mp4", writer='pillow', fps=30)
                print("Animation saved as MP4 using pillow writer")
            except Exception as e2:
                print(f"Both MP4 methods failed: {e2}")
                # Fallback to GIF
                try:
                    animation.save("scaling_simulation_animation.gif", writer='pillow', fps=30)
                    print("Animation saved as GIF instead")
                except Exception as e3:
                    print(f"All save methods failed: {e3}")
        
        # type(animation)
        # input()
    # else:
    #     scenario1.run(10)
    #     res = None
    # res