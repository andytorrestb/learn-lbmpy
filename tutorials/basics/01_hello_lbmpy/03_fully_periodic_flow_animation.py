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
- 'fully_periodic_flow.png': Static plot of the final velocity field.
- 'fully_periodic_flow_animation.gif': Animated GIF of the velocity field evolution.
- 'fully_periodic_flow_animation.mp4': Animated MP4 of the velocity field evolution (if ffmpeg is available).

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
# import matplotlib.pyplot as plt  # Explicit import for standard matplotlib functions
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
ldc = create_fully_periodic_flow(initial_velocity=init_vel, relaxation_rate=1.97)


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
plt.savefig("fully_periodic_flow.png")
plt.close()
print("Static plot saved as 'fully_periodic_flow.png'")

# Create animation using native plt.vector_field function (additional functionality)
print("Creating animation with native vector_field function...")

# SOLUTION: Force pyplot to manage the figure throughout the animation
# Instead of using plt.subplots(), use plt.figure() which stays managed
print("Using pyplot approach with lbmpy's native plt.vector_field()...")

# Close any existing figures first
plt.close('all')

# Create figure using pyplot's figure() method to ensure it stays managed
fig = plt.figure(figsize=(10, 8), dpi=100)
ax = fig.add_subplot(111)

# Force pyplot to register this figure (important!)
# plt.show(block=False)
# plt.pause(0.001)  # Brief pause to ensure registration

# ==========================================================================================
# ||                       4) Create animation(s) of results                              ||
# ==========================================================================================

import warnings

def animate_frame_pyplot(frame_idx):
    """Animation function using lbmpy's native plt.vector_field() method - ULTIMATE FIX"""
    try:
        vel_field = velocity_data[frame_idx]
        if vel_field is None:
            return
        
        # ULTIMATE SOLUTION: Clear everything and recreate from scratch
        # This avoids all pyplot state management issues
        plt.clf()  # Clear the entire figure
        
        # Now plt.vector_field will work because we have a clean slate
        plt.vector_field(vel_field, step=3)
        plt.title(f"Fully periodic flow - Step {time_steps[frame_idx]}")
        plt.xlabel("x")
        plt.ylabel("y")
        
        # Return the current axes so animation system knows what to save
        return plt.gca()
        
    except Exception as e:
        print(f"ERROR in animate_frame({frame_idx}): {e}")
        # Even if lbmpy fails, provide a clean axes
        plt.clf()
        plt.text(0.5, 0.5, f"Frame {frame_idx} failed: {str(e)[:30]}", 
               ha='center', va='center', transform=plt.gca().transAxes)
        return plt.gca()

def animate_frame_direct(frame_idx):
    """Animation function for direct matplotlib approach (fallback)"""
    try:
        ax.clear()
        
        vel_field = velocity_data[frame_idx]
        if vel_field is None:
            return
            
        # Fallback: use matplotlib quiver if lbmpy method fails
        if vel_field.ndim == 3 and vel_field.shape[2] >= 2:
            step = 3
            y, x = np.mgrid[0:vel_field.shape[0]:step, 0:vel_field.shape[1]:step]
            u = vel_field[::step, ::step, 0]
            v = vel_field[::step, ::step, 1]
            ax.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1, alpha=0.7)
        
        ax.set_title(f"Fully periodic flow - Step {time_steps[frame_idx]}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
    except Exception as e:
        print(f"ERROR in direct animate_frame({frame_idx}): {e}")

# Use the lbmpy native approach with the ultimate fix
animate_frame = animate_frame_pyplot

# Create animation object - NOTE: we don't need the ax parameter anymore since we use plt.clf()
print(f"Creating animation with {len(velocity_data)} frames...")

# Debug: Check the data before creating animation
if len(velocity_data) == 0:
    print("ERROR: No velocity data collected! Animation cannot be created.")
    exit(1)

print(f"First velocity field shape: {velocity_data[0].shape}")
print(f"First velocity field dtype: {velocity_data[0].dtype}")
print(f"Time steps collected: {len(time_steps)}")
print(f"Sample time steps: {time_steps[:5]} ... {time_steps[-5:]}")

# Examine the structure of velocity data
sample_vel = velocity_data[0]
print(f"Velocity data dimensions: {sample_vel.ndim}")
print(f"Velocity data shape: {sample_vel.shape}")
if sample_vel.ndim == 3:
    print(f"Shape breakdown: height={sample_vel.shape[0]}, width={sample_vel.shape[1]}, components={sample_vel.shape[2]}")
elif sample_vel.ndim == 2:
    print("2D array - might be magnitude only")

# Check if any velocity data is problematic
for i, vel_field in enumerate(velocity_data[:5]):  # Check first 5 frames
    if vel_field is None:
        print(f"ERROR: Frame {i} is None!")
    elif vel_field.size == 0:
        print(f"ERROR: Frame {i} is empty!")
    elif not isinstance(vel_field, np.ndarray):
        print(f"ERROR: Frame {i} is not a numpy array: {type(vel_field)}")
    else:
        print(f"Frame {i} shape: {vel_field.shape}, dtype: {vel_field.dtype}, range: [{np.min(vel_field):.6f}, {np.max(vel_field):.6f}]")

# Create the animation using the figure (plt.clf() approach doesn't need axes parameter)
anim = animation.FuncAnimation(fig, animate_frame, frames=len(velocity_data),
                              interval=200, blit=False, repeat=True)

# Test the animation function before saving
print("Testing animation function...")
try:
    # Test a few frames to make sure the approach works
    animate_frame(0)  # Test first frame
    print("First frame animated successfully")
    
    if len(velocity_data) > 1:
        animate_frame(len(velocity_data)//2)  # Test middle frame
        print("Middle frame animated successfully")
        
        animate_frame(len(velocity_data)-1)  # Test last frame  
        print("Last frame animated successfully")
        
    print("Animation function test completed successfully!")
    
except Exception as e:
    print(f"ERROR in animate_frame function: {e}")
    import traceback
    traceback.print_exc()
    print("Animation function has issues, but proceeding with save attempt...")

# frames per second
fps = 60

# Save animation as GIF
print("Saving animation as GIF...")
try:
    print(f"Attempting to save {len(velocity_data)} frames at {fps} fps...")
    print(f"Animation object type: {type(anim)}")
    
    # Try to save with more error detail
    anim.save("fully_periodic_flow_animation_native.gif", writer='pillow', fps=fps)
    print("Animation saved as 'fully_periodic_flow_animation_native.gif'")
except IndexError as e:
    print(f"IndexError (list index out of range): {e}")
    print("This usually means:")
    print("  1. Empty velocity_data list")
    print("  2. Frame indexing issue in animate_frame function") 
    print("  3. Issue with plt.vector_field accessing data")
    
    # Try alternative approach - create animation with simpler method
    print("Trying alternative animation approach...")
    try:
        # Close current problematic figure
        plt.close(fig)
        
        # Create a new figure with simpler approach
        fig_alt = plt.figure(figsize=(10, 8))
        
        def animate_simple(frame_idx):
            plt.clf()  # Clear entire figure
            plt.vector_field(velocity_data[frame_idx], step=3)
            plt.title(f"Fully periodic flow - Step {time_steps[frame_idx]}")
            plt.xlabel("x")
            plt.ylabel("y")
        
        # Create new animation
        anim_alt = animation.FuncAnimation(fig_alt, animate_simple, frames=len(velocity_data),
                                          interval=200, blit=False, repeat=True)
        
        # Try saving the alternative animation
        anim_alt.save("fully_periodic_flow_animation_alt.gif", writer='pillow', fps=fps)
        print("Alternative animation saved as 'fully_periodic_flow_animation_alt.gif'")
        plt.close(fig_alt)
        
    except Exception as e_alt:
        print(f"Alternative approach also failed: {e_alt}")
        import traceback
        traceback.print_exc()
    
    # Additional debugging
    import traceback
    print("\nOriginal error traceback:")
    traceback.print_exc()
    
except Exception as e:
    print(f"Other error saving GIF: {type(e).__name__}: {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()

# Save animation as MP4 (requires ffmpeg)
print("Attempting to save animation as MP4...")
try:
    # Method 1: Try with explicit figure management
    plt.figure(fig.number)  # Make sure pyplot knows about our figure
    anim.save("fully_periodic_flow_animation_native.mp4", writer='ffmpeg', fps=fps, bitrate=1800)
    print("Animation saved as 'fully_periodic_flow_animation_native.mp4'")
except Exception as e:
    print(f"Method 1 failed: {e}")

    # Method 2: Try saving with different approach
    try:
        # Use the figure's savefig method directly through animation
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, bitrate=1800)
        anim.save("fully_periodic_flow_animation.mp4", writer=writer)
        print("Animation saved as 'fully_periodic_flow_animation_native.mp4' (Method 2)")
    except Exception as e2:
        print(f"Method 2 also failed: {e2}")

        # Method 3: Alternative figure creation approach with proper axes handling
        print("Trying alternative figure management...")
        try:
            # Recreate with pyplot figure but use axes-based approach
            plt.close('all')  # Close existing figures
            fig2 = plt.figure( dpi=100)
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
            anim2.save("fully_periodic_flow_animation.mp4", writer='ffmpeg', fps=5, bitrate=1800)
            print("Animation saved as 'fully_periodic_flow_animation.mp4' (Method 3 - Warning-Free)")
            plt.close(fig2)
        except Exception as e3:
            print(f"All methods failed. Final error: {e3}")
            print("GIF saved successfully, but MP4 export has issues with figure management.")

plt.close(fig)
print("Animation creation complete!")
