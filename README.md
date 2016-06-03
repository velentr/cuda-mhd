# CUDA MHD

GPU-accelerated magnetohydrodynamics simulation using CUDA. Notes:
* Uses Classical Runge-Kutta for time propagation for second-order error
* Implements ideal MHD equations for regular spatial grid

Possible improvements:
* Use struct instead of array of doubles as better abstraction
