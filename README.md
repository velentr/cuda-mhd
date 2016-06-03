# CUDA MHD

GPU-accelerated magnetohydrodynamics simulation using CUDA. Notes:

- Uses Classical Runge-Kutta for time propagation for second-order error
- Implements ideal MHD equations for regular spatial grid

## Usage

To run the CPU version, go to the `cpu` directory and run `make mhd`. To run the
program, saving output to a file, run:

```
mhd > output.dat
```

To run the GPU version, go to the `gpu` directory and run `make mhd`. To run the
simulation:

```
mhd <threads_per_block> <num_blocks> > output-gpu.dat
```

## Program Explanation

Magnetohydrodynamics (MHD) is the study of fluid flow subject to changing
magnetic fields. Its diverse range of applications range from laboratory
experiments to solar phenomena to supernovae, the spectacular deaths of stars
hundreds of times the size of our Sun. MHD is described by a set of highly
complex nonlinear partial differential equations that require numerical
simulations of high accuracy to capture all interesting phenomena, a
computationally expensive task. Nevertheless, the numerical solution is highly
parallelizable and is well-suited to acceleration via GPU computation.

Our code performs a numerical MHD simulation alongside a sample simulation in
which a set of realistic initial conditions is simulated. The
code is capable of modelling any phenomena with appropriate choice of
initial conditions.

## Expected Results

The results printed out from the program gives the state of the physical system
at different timesteps. The physical state of the system in all components of
the state vector at each point in the grid observed from our program agree with
analytical expectation. Moreover, both the CPU and the GPU implementations agree
for the timescales that we are looking at.

## Analysis of Performance

Currently, the GPU implementation is slightly slower than the CPU version. This
is likely due to the scale of the simulation: if we were to run for many more
timesteps, the overhead from copying data to and from the GPU would be less
significant. Unfortunately, double-precision floats are not sufficient to
simulate more timesteps without a significant amount of additional numerical debugging
as results begin to differ between the GPU and
the CPU---likely because one FPU is IEEE compliant and one is not.

