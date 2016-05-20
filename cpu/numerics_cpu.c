/* Author: Yubo Su
 * Numerics suite for cuda-mhd CPU implementation
 */
#include <stdlib.h>
#include <string.h>

void step(void (*f)(float*, float*, int), float dt, float* yn,
        float* ynew, float* dy, int leny)
/* Given some dy/dt = f(y), compute the dy for a given yn and store into yn +
 * dy into ynew. Uses Forward Euler for now
 *
 * Input:
 *  void (*f)           : computes dy/dt = f(y), stores into second float*
 *  float dt            : timestep dt
 *  float* yn           : current y
 *  float* ynew         : new y
 *  float* dy           : get return from (*f) (malloc in caller)
 *  int leny            : length of y vector
 */
{
    int i;
    (*f)(yn, dy, leny); /* compute dy/dt into dy */
    for (i = 0; i < leny; i++) 
    {
        ynew[i] = yn[i] + dy[i] * dt;
    }
}

float* solve(void (*f)(float*, float*, int),
        float* y0, float dt, int nsteps, int leny)
/* Given some initial y(t = 0) and dy/dt = f(y) (autonomous ODE), computes
 * y(nsteps * dt) by iterating y_n -> y_{n+1} via step. y assumed in
 * generality to be a vector
 *
 * Input:
 *  void (*f)           : computes dy/dt = f(y), stores into second float*
 *  float* init         : y(t=0)
 *  float dt            : dt timestep
 *  int nsteps          : number of timesteps to evolve
 *  int leny            : length of y vector
 * Output:
 *  float*              : (nsteps + 1) * leny vector containing y(t), contains
 *                        timesteps dt * [0,nsteps] inclusive (hence +1)
 */
{
    /* stores total trajectory */
    float* dy = (float*) malloc(leny * sizeof(float));
    float* ret = (float*) malloc((nsteps + 1) * leny * sizeof(float));
    int i;

    /* start with ret = y0 */
    memcpy(ret, y0, leny * sizeof(float));

    /* at each step, get ynew into ret */
    for (i = 0; i < nsteps; i++) 
    {
        step(f, dt, ret + i * leny, ret + (i + 1) * leny, dy, leny);
    }

    /* cleanup */
    free(dy);
    return ret;
}
