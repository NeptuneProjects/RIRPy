# 🔊RIRPy: Room Impulse Response Modeling

By William Jenkins  
Scripps Institution of Oceanography  
UC San Diego

This package provides a set of functions for simulating room impulse responses (RIRs) in rectangular spaces using the method of images.
The package is a direct port of MATLAB code written by [Hayden Johnson](https://github.com/haydenallenjohnson/modelling_tank_wall_reflections), which was originally used for simulating the impulse response inside a tank for underwater acoustics research. Hayden and coauthors discuss the validation of their code in their paper:
> Shayna Grossman, Hayden Johnson, Dale Stokes, Grant Deane; Toward monitoring submarine glacier melt using hydroacoustics: The role of timescale in the signal of bubble release. J. Acoust. Soc. Am. 1 September 2024; 156 (3): 1820–1838. [https://doi.org/10.1121/10.0028628](https://doi.org/10.1121/10.0028628)

The method of images is used to calculate the various possible reflection paths, and is possible under the assumption that the speed of sound is constant in the medium.
For non-constant sound speed, ray tracing methods would be more appropriate.

The "walls" and "floor" of the space are assumed to have the same reflection coefficient.
To account for the pressure release boundary condition between water and air, the "surface" (or ceiling) is assumed to have its own reflection coefficient.
In cases where the boundaries consist of a rigid surface, such as the walls of a tank, the reflection coefficient should be set to a positive value.
For a pressure release boundary conditions, the reflection coefficient should be set to a negative value.

The computation of the images is based on the forumlation presented in:
>Jont B. Allen, David A. Berkley; Image method for efficiently simulating small‐room acoustics. J. Acoust. Soc. Am. 1 April 1979; 65 (4): 943–950. [https://doi.org/10.1121/1.382599](https://doi.org/10.1121/1.382599)


## Installation

`RIRPy` can be installed using `pip`:
```bash
pip install git+https://github.com/NeptuneProjects/RIRPy.git
```

## Usage

`RIRPy` provides a simple, high-level interface for simulating room impulse responses.
Users must specify the dimensions of the room, the position of the source and receiver, and the reflection coefficients for the walls and ceiling (surface).
The general procedure is as follows:
1. Define the propagation environment and geometry, source and receiver positions, and reflection coefficients.
2. Define a source signal, which can be a simple impulse or any other signal.
3. Propagate the source signal through the room using the `simulate_rir` function.
4. The output will be the source signal convolved with the room impulse response, resulting in the simulated received signal.

Cylindrical spreading is applied to the impulse response for each reflection.
This is not necessarily accurate in all cases, but is likely an acceptable approximation for most scenarios.

A demonstration of the package can be found in the `examples` directory.

## Performance Considerations

The method of images can be computationally expensive, depending on the room geometry and number of reflections.
To mitigate this, just-in-time compilation with `Numba` is used to accelerate computation of the impulse response.
Parameters that particularly affect computation time include:
- `cutoff_time`: The maximum time for which reflections are calculated. The longer the cutoff time, the more reflections are calculated, and the longer the computation time.
- `length_x`, `length_y`, `length_z`: The dimensions of the room. Somewhat counterintuitively, decreasing the room size while keeping `cutoff_time` fixed will take longer to compute due to the increased number of reflections.
