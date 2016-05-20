/* Author: Yubo Su
 * Numerics suite for cuda-mhd CPU implementation
 *
 * TODO define overloads for step() and solve() for single float,
 *      maybe for multiple function pointers? So can do multiple steps per time
 *      step
 */
#ifndef NUMERICS_CPU_H

#define NUMERICS_CPU_H
#include <stdlib.h>

void step(void (*f)(float*, float*, int), float dt, float* yn,
        float* ynew, float* dy, int leny);
/* Given some dy/dt = f(y), compute the dy for a given yn and store into dy 
 *
 * Input:
 *  void (*f)           : computes dy/dt = f(y), stores into second float*
 *  float dt            : timestep dt
 *  float* yn           : current y
 *  float* ynew         : new y
 *  float* dy           : get return from (*f) (malloc in caller)
 *  int leny            : length of y vector
 */

float* solve(void (*f)(float*, float*, int),
        float* y0, float dt, int nsteps, int leny);
/* Given some initial y(t = 0) and dy/dt = f(y) (autonomous ODE), computes
 * y(nsteps * dt) by iterating y_n -> y_{n+1} via Runge-Kutta. y assumed in
 * generality to be a vector
 *
 * Input:
 *  void (*f)           : computes dy/dt = f(y), stores into second float*
 *  float* init         : y(t=0)
 *  float dt            : dt timestep
 *  int nsteps          : number of timesteps to evolve
 *  int leny            : length of y vector
 * Output:
 *  float*              : nsteps * leny vector containing y(t)
 * */

#endif /* NUMERICS_CPU_H*/
