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
        float* ynew, float* dy, int leny, float* scratch, float* scratch2);
/* Given some dy/dt = f(y), compute the dy for a given yn and store into yn +
 * dy into ynew. Uses Runge-Kutta:
 * k1 = f(y), k2 = f(y + k1 * dt/2), k3 = f(y + k2 * dt/2), 
 *      k4 = f(y + k3 * dt), ynew = yn + (k1 + 2k2 + 2k3 + k4)/6 * dt
 * 
 * 2 scratch vectors required to not nuke in case ynew = yn (overwrite)
 *
 * Input:
 *  void (*f)           : computes dy/dt = f(y), stores into second float*
 *  float dt            : timestep dt
 *  float* yn           : current y
 *  float* ynew         : new y
 *  float* dy           : get return from (*f) (malloc in caller)
 *  int leny            : length of y vector
 *  float* scratch[2]   : scratch vector, simply to avoid re-mallocing,
 *                          also leny
 */

float* solve(void (*f)(float*, float*, int),
        void (*g)(float*, int),
        float* y0, float dt, int nsteps, int leny, int save_skip);
/* Given some initial y(t = 0) and dy/dt = f(y) (autonomous ODE), computes
 * y(nsteps * dt) by iterating y_n -> y_{n+1} via step. y assumed in
 * generality to be a vector
 *
 * Input:
 *  void (*f)           : computes dy/dt = f(y), stores into second float*
 *  void (*g)           : any per-step processing (compute temp values, div B)
                        : can pass NULL to bypass this processing
 *  float* init         : y(t=0)
 *  float dt            : dt timestep
 *  int nsteps          : number of timesteps to evolve
 *  int leny            : length of y vector
 *  int save_skip       : every save_skip timesteps, save trajectory
 * Output:
 *  float*              : (nsteps + 1) * leny vector containing y(t), contains
 *                        timesteps dt * [0,nsteps] inclusive (hence +1)
 */

#endif /* NUMERICS_CPU_H*/
