# learn-lbmpy

## Introduction

This repository contains a collection of tutorial examples and solutions for learning the Lattice Boltzmann Method (LBM) using the `lbmpy` framework. This repository is meant to mirror the examples covered in the official documentation (with additional context and functionality). To achieve this, the examples covered progress from basic "Hello World" simulations to advanced multiphase and turbulence modeling cases. This is still a work in progress. Contributions are welcome. 

For detailed information about the `lbmpy` framework itself, please refer to the [official lbmpy documentation](https://pycodegen.pages.i10git.cs.fau.de/lbmpy/) and [source code](https://github.com/lssfau/lbmpy).

## Repository Structure

NOTE: Only basics and turbulence modeling are covered as of now. Contributions are welcome here.

```
tutorials/
├── basics/           # Fundamental LBM concepts and workflows
├── multiphase/       # Multi-phase flow simulations
├── nonnewtonian/     # Non-Newtonian fluid behavior
├── thermal/          # Thermal effects in LBM
├── thermocapillary/  # Surface tension phenomena
└── turbulence/       # Turbulence modeling and LES
```

## Test Cases Overview

### Basics (`tutorials/basics/`)

A sequential learning path with 6 progressive modules:

#### 0. **lbmpy Overview** (`00_lbmpy_overview/`)
*This section includes test cases covered in the documentation overview [See Link](http://pycodegen.pages.i10git.cs.fau.de/lbmpy/notebooks/00_tutorial_lbmpy_walberla_overview.html).*
- **Lid-driven cavity flow**: Classic benchmark for incompressible flow
- **Fully periodic flow**: Shear layer instability demonstration
- **Animation examples**: Time-evolution visualization techniques
- **Key Learning**: Basic lbmpy workflow and visualization

#### 1. **Hello lbmpy** (`01_hello_lbmpy/`)
*This section introduces basic LBM workflows using pre-configured scenarios [See Link](http://pycodegen.pages.i10git.cs.fau.de/lbmpy/notebooks/01_tutorial_hello_lbmpy.html).*
- **01_lid_driven_cavity.py**: 2D and 3D cavity flow with varying relaxation rates
- **02_fully_periodic_flow.py**: Shear layer development and vorticity analysis
- **03_fully_periodic_flow_animation.py**: Advanced animation techniques
- **04_channel_flow.py**: Poiseuille flow with obstacles
- **Key Learning**: Pre-configured scenarios, GPU/CPU execution, result visualization

#### 2. **Geometry and Boundary Conditions** (`02_geom_and_bcs/`)
*This section covers basic methods to create simulation domains and set up boundary conditions [See Link](https://pycodegen.pages.i10git.cs.fau.de/lbmpy/notebooks/02_tutorial_boundary_setup.html).*
- **01_geometry.py**: Complex domain creation and geometric masks
- **02_boundary_conditions.py**: 3D pipe flow with inflow/outflow/wall boundaries
- **Key Learning**: Boundary condition setup, geometric callback functions, 3D simulations

#### 3. **Defining LBM Methods** (`03_defining_lbm_methods/`)
*This section explores different LBM collision models and their effects [See Link](https://pycodegen.pages.i10git.cs.fau.de/lbmpy/notebooks/03_tutorial_lbm_formulation.html).*
- **01_lbm_method.py**: Comprehensive comparison of collision models
  - SRT (Single Relaxation Time)
  - MRT (Multiple Relaxation Time) - weighted and orthogonal
  - Central Moment methods
  - Custom moment definitions
- **Key Learning**: Method selection, relaxation rate effects, moment spaces

#### 4. **Cumulant LBM** (`04_cumulant_lbm/`)
*Apply advanced cumulant-based LBM for challenging flow cases [See Link](https://pycodegen.pages.i10git.cs.fau.de/lbmpy/notebooks/04_tutorial_cumulant_LBM.html).*
- **01_cumulant_lbm.py**: High Reynolds number flow around obstacles
  - Manual kernel creation and optimization
  - Complex boundary handling
  - Animation export (MP4/GIF)
- **Key Learning**: Low-level lbmpy usage, cumulant methods, high-Re flows

#### 5. **Non-dimensionalization and Scaling** (`05_non_dim_and_scaling/`)
*Convert physical parameters to simulation units and analyze scaling [See Link](https://pycodegen.pages.i10git.cs.fau.de/lbmpy/notebooks/05_tutorial_nondimensionalization_and_scaling.html).*
- **01_scaling.py**: Physical-to-lattice unit conversion
  - Reynolds number scaling
  - Diffusive vs. acoustic scaling
  - Parameter sensitivity analysis
- **Key Learning**: Physical parameter mapping, dimensional analysis

### Advanced Topics
#### 6. **Turbulence** (`tutorials/turbulence/`)
*Covers the basics of developing turbulence models for LBM [See Link](https://pycodegen.pages.i10git.cs.fau.de/lbmpy/notebooks/06_tutorial_modifying_method_smagorinsky.html).*
- **06_smagorinsky.py**: Large Eddy Simulation (LES) with Smagorinsky model
- Subgrid-scale modeling
- Turbulent channel flow

NOTE: All subsecuent tutorials need to be developed for this repository.

#### **Multiphase** (`tutorials/multiphase/`)
- Multi-component flow simulations
- Phase separation and interface dynamics
- Contact angle effects

#### **Non-Newtonian** (`tutorials/nonnewtonian/`)
- Shear-thinning and shear-thickening fluids
- Viscoelastic flow behavior
- Rheological model implementation

#### **Thermal** (`tutorials/thermal/`)
- Temperature-coupled LBM
- Natural convection
- Heat transfer applications

#### **Thermocapillary** (`tutorials/thermocapillary/`)
- Surface tension modeling
- Marangoni effects
- Droplet dynamics

## Getting Started
1. **Environment Setup**: The repository demands several Python packages. The easiest way to manage this is with a virtual environment. Requirements are then stored in the requirements.txt file.

    **Example: Creating a virtual environment**
    ```bash
    python -m venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

2. **Running Examples**: Navigate to any tutorial folder and execute the Python scripts:
   ```bash
   cd tutorials/basics/01_hello_lbmpy/
   python 01_lid_driven_cavity.py
   ```

3. **Learning Path**: Follow the sequential structure in `tutorials/basics/` for systematic learning.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

## Further Information

For comprehensive documentation, theoretical background, and advanced features of the `lbmpy` framework, please visit:
- [lbmpy Documentation](https://pycodegen.pages.i10git.cs.fau.de/lbmpy/)
- [lbmpy GitHub Repository](https://github.com/lssfau/lbmpy)
- [pystencils Documentation](https://pycodegen.pages.i10git.cs.fau.de/pystencils/)
