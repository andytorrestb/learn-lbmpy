"""
Hello LBMPY (Animated)

This script runs a lid-driven cavity flow simulation using lbmpy and pystencils. It is meant to serve as a "Hello World" example for lbmpy.
This case illustrates how a user can run a pre-configured case in lbmpy and extract data. The original exercise a static plot of the results
and this script adds an animation by collecting velocity field data at set time intervals to serve as frames in the video.

Main functionalities and features:
- Initializes a lid-driven cavity simulation with specified domain size and LBM configuration.
- Supports both CPU and GPU execution (if cupy is available).
- Runs the simulation for a set number of steps, periodically storing velocity field snapshots.
- Creates a static plot of the final velocity field using lbmpy's native vector_field plotting.
- Generates an animation of the velocity field evolution using matplotlib's FuncAnimation and lbmpy's vector_field.
- Saves the animation as both GIF and MP4 formats (MP4 requires ffmpeg).
- Handles exceptions during animation saving and provides informative output.

Dependencies:
- pystencils
- lbmpy
- matplotlib
- cupy (optional, for GPU support)
- pillow (for GIF saving)
- ffmpeg (for MP4 saving)

Input Paramters (within the script):
- total_steps: 500
- save_interval: 10
- domain_size: (100, 100)
- relaxation_rate: 1.6


Output files:
- 'lid_driven_cavity.png': Static plot of the final velocity field.
- 'lid_driven_cavity_animation.gif': Animated GIF of the velocity field evolution.
- 'lid_driven_cavity_animation.mp4': Animated MP4 of the velocity field evolution (if ffmpeg is available).

Author: Andy Torres
Source of Original Example: https://pycodegen.pages.i10git.cs.fau.de/lbmpy/notebooks/00_tutorial_lbmpy_walberla_overview.html

"""

# ==========================================================================================
# ||                     1) Configure enviorment and run simulation                       ||
# ==========================================================================================

# Import libraries as necessary.
import pystencils as ps
from pystencils import Target, CreateKernelConfig
from lbmpy.session import *  # This provides plt.vector_field()
import matplotlib.animation as animation
import numpy as np

# Import GPU functionality (if available)
try:
    import cupy
except ImportError:
    cupy = None
    gpu = False
    target = ps.Target.CPU

if cupy:
    gpu = True
    target = ps.Target.GPU

# Create the lid-driven cavity simulation
ldc = create_lid_driven_cavity(domain_size=(100, 100),
                               lbm_config=LBMConfig(method=Method.SRT, relaxation_rate=1.6))

# Animation parameters
total_steps = 500
save_interval = 1 # Save data for every step of animation
velocity_data = []
time_steps = []

print("Running simulation and collecting data for animation...")

# Run simulation step by step and collect velocity data
for step in range(total_steps + 1):
    if step % save_interval == 0:
        # Store velocity field for animation
        velocity_data.append(ldc.velocity_slice().copy())
        time_steps.append(step)
        print(f"Collected data at step {step}")

    if step < total_steps:
        ldc.run(1)  # Run one step at a time

# ==========================================================================================
# ||                     2) Check results for invalid data                                ||
# ==========================================================================================

# After collecting velocity data, add inspection
print("Inspecting collected velocity data for NaN or Inf values...")
any_nan = False
for i, vel_field in enumerate(velocity_data):
    # Check for NaN values
    nan_count = np.isnan(vel_field).sum()
    # Check for infinite values
    inf_count = np.isinf(vel_field).sum()
    # Check for zero values
    zero_count = (vel_field == 0).sum()

    if nan_count > 0 or inf_count > 0:
        print(f"Frame {i}: NaN={nan_count}, Inf={inf_count}, Zero={zero_count}")
        print(f"  Min: {np.nanmin(vel_field)}, Max: {np.nanmax(vel_field)}")
        any_nan = True

if any_nan:
    print("Warning: Some frames contain NaN or Inf values!")
else:
    print("All frames are valid.")

# Test each frame individually to find the problematic one
print("Testing individual frames for rendering issues...")
for i, vel_field in enumerate(velocity_data):
    try:
        plt.figure()
        plt.vector_field(vel_field, step=3)
        plt.close()
        if i % 50 == 0:  # Print progress
            print(f"Frame {i} rendered successfully")
    except Exception as e:
        print(f"Frame {i} caused error: {e}")

# ==========================================================================================
# ||                      3) Create static figure of results                              ||
# ==========================================================================================

# Create static plot (original functionality)
plt.figure(dpi=200)
plt.vector_field(ldc.velocity_slice(), step=3)
plt.title("Lid-driven cavity flow (Final State)")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("lid_driven_cavity.png")
plt.close()
print("Static plot saved as 'lid_driven_cavity.png'")

# Create animation using native plt.vector_field function (additional functionality)
print("Creating animation with native vector_field function...")

# Option A: Use subplots with explicit axes clearing
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

# Ensure the figure is properly managed by pyplot
plt.figure(fig.number)

# ==========================================================================================
# ||                       4) Create animation(s) of results                              ||
# ==========================================================================================

import warnings
def animate_frame(frame_idx):
    """Animation function for each frame using explicit axes clearing"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Clear specific axes instead of entire figure
        ax.clear()

        # Set the current axes to our subplot axes before calling plt.vector_field
        plt.sca(ax)

        # Use the native lbmpy vector_field plotting function
        plt.vector_field(velocity_data[frame_idx], step=3)
        plt.title(f"Lid-driven cavity flow - Step {time_steps[frame_idx]}")
        plt.xlabel("x")
        plt.ylabel("y")

        if w:
            print(f"Frame {frame_idx} caused {len(w)} warning(s): {[str(warning.message) for warning in w]}")


# Create animation object
print(f"Creating animation with {len(velocity_data)} frames...")
anim = animation.FuncAnimation(fig, animate_frame, frames=len(velocity_data),
                              interval=200, blit=False, repeat=True)

# frames per second
fps = 60

# Save animation as GIF
print("Saving animation as GIF...")
try:
    anim.save("lid_driven_cavity_animation_native.gif", writer='pillow', fps=fps)
    print("Animation saved as 'lid_driven_cavity_animation_native.gif'")
except Exception as e:
    print(f"Error saving GIF: {e}")

# Save animation as MP4 (requires ffmpeg)
print("Attempting to save animation as MP4...")
try:
    # Method 1: Try with explicit figure management
    plt.figure(fig.number)  # Make sure pyplot knows about our figure
    anim.save("lid_driven_cavity_animation_native.mp4", writer='ffmpeg', fps=fps, bitrate=1800)
    print("Animation saved as 'lid_driven_cavity_animation_native.mp4'")
except Exception as e:
    print(f"Method 1 failed: {e}")

    # Method 2: Try saving with different approach
    try:
        # Use the figure's savefig method directly through animation
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, bitrate=1800)
        anim.save("lid_driven_cavity_animation.mp4", writer=writer)
        print("Animation saved as 'lid_driven_cavity_animation_native.mp4' (Method 2)")
    except Exception as e2:
        print(f"Method 2 also failed: {e2}")

        # Method 3: Alternative figure creation approach with proper axes handling
        print("Trying alternative figure management...")
        try:
            # Recreate with pyplot figure but use axes-based approach
            plt.close('all')  # Close existing figures
            fig2 = plt.figure(figsize=(10, 8), dpi=100)
            ax2 = fig2.add_subplot(111)  # Create axes explicitly

            import warnings
            def animate_frame_v2(frame_idx):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Clear axes instead of entire figure
                    ax2.clear()

                    # Set current axes
                    plt.sca(ax2)

                    # Plot using lbmpy's vector_field
                    plt.vector_field(velocity_data[frame_idx], step=3)
                    plt.title(f"Lid-driven cavity flow - Step {time_steps[frame_idx]}")
                    plt.xlabel("x")
                    plt.ylabel("y")

                    if w:
                        print(f"Frame {frame_idx} caused {len(w)} warning(s): {[str(warning.message) for warning in w]}")

            anim2 = animation.FuncAnimation(fig2, animate_frame_v2, frames=len(velocity_data),
                                          interval=200, blit=False, repeat=True)
            anim2.save("lid_driven_cavity_animation.mp4", writer='ffmpeg', fps=5, bitrate=1800)
            print("Animation saved as 'lid_driven_cavity_animation.mp4' (Method 3 - Warning-Free)")
            plt.close(fig2)
        except Exception as e3:
            print(f"All methods failed. Final error: {e3}")
            print("GIF saved successfully, but MP4 export has issues with figure management.")

plt.close(fig)
print("Animation creation complete!")
