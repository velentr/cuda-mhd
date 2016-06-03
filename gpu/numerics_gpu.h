/* Author: Yubo Su
 * Numerics suite for cuda-mhd CPU implementation
 *
 * TODO define overloads for step() and solve() for single double,
 *      maybe for multiple function pointers? So can do multiple steps per time
 *      step
 */
#ifndef NUMERICS_CPU_H
#define NUMERICS_CPU_H


#include <stdlib.h>

#define NUM_COMPS 12
#define U1(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)))
#define U2(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 1)
#define U3(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 2)
#define U4(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 3)
#define U5(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 4)
#define U6(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 5)
#define U7(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 6)
#define U8(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 7)
#define U9(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 8)
#define U10(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 9)
#define U11(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 10)
#define U12(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 11)
#define MU 1
#define DL 1000
#define GAMMA 1.4

double *solve(double *y0, double dt, int nsteps, int leny, int save_skip,
        size_t nblk, size_t thr);
/* Given some initial y(t = 0) and dy/dt = f(y) (autonomous ODE), computes
 * y(nsteps * dt) by iterating y_n -> y_{n+1} via step. y assumed in
 * generality to be a vector
 *
 * Input:
 *  void (*f)           : computes dy/dt = f(y), stores into second double*
 *  void (*g)           : any per-step processing (compute temp values, div B)
                        : can pass NULL to bypass this processing
 *  double* init         : y(t=0)
 *  double dt            : dt timestep
 *  int nsteps          : number of timesteps to evolve
 *  int leny            : length of y vector
 *  int save_skip       : every save_skip timesteps, save trajectory
 * Output:
 *  double*              : (nsteps + 1) * leny vector containing y(t), contains
 *                        timesteps dt * [0,nsteps] inclusive (hence +1)
 */

#endif /* NUMERICS_CPU_H*/
