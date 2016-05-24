/* Author: Yubo Su
 * Numerics suite for cuda-mhd CPU implementation
 */
#include <stdlib.h>
#include <string.h>
#define MIN(a, b) (a) < (b) ? (a) : (b)
#define MAX(a, b) (a) > (b) ? (a) : (b)

void step(void (*f)(float*, float*, int), float dt, float* yn,
        float* ynew, float* dy, int leny, float* scratch, float* scratch2)
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
{
    /* organization scheme:
     * yn = y
     * scratch2 = k1 + ... (running tally)
     * scratch = y + ...
     * dy = f(y + ...)
     *
     * then copy correct value into ynew */
    int i;

    (*f)(yn, dy, leny); /* compute k1 into dy */
    /* scratch = yn + k1 * dt/2 */
    for (i = 0; i < leny; i++) 
    {
        scratch[i] = yn[i] + dy[i] * dt / 2; /* y + k1 * dt/2 */
        scratch2[i] = dy[i] / 6; /* k1 / 6 */
    }

    (*f)(scratch, dy, leny); /* compute k2 into dy */
    for (i = 0; i < leny; i++) 
    {
        scratch[i] = yn[i] + dy[i] * dt / 2; /* y + k2 * dt/2 */
        scratch2[i] += dy[i] / 3; /* (k1 + 2k2) / 6 */
    }

    (*f)(scratch, dy, leny); /* compute k3 into dy */
    for (i = 0; i < leny; i++) 
    {
        scratch[i] = yn[i] + dy[i] * dt; /* y + k3 * dt */
        scratch2[i] += dy[i] / 3; /* (k1 + 2k2 + 2k3) / 6 */
    }

    (*f)(scratch, dy, leny); /* compute k4 into dy */
    for (i = 0; i < leny; i++) 
    {
        /* dt * (k1 + 2k2 + 2k3 + k4) / 6 + yn */
        ynew[i] = dt * (scratch2[i] + dy[i] / 6) + yn[i];
    }
}

float* solve(void (*f)(float*, float*, int),
        void (*g)(float*, int),
        float* y0, float dt, int nsteps, int leny, int save_skip)
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
                        : use only divisible by nsteps please
 * Output:
 *  float*              : (nsteps + 1) * leny vector containing y(t), contains
 *                        timesteps dt * [0,nsteps / saveskip]
 */
{
    /* stores total trajectory */
    float* dy = (float*) malloc(leny * sizeof(float));
    float* scratch = (float*) malloc(leny * sizeof(float));
    float* scratch2 = (float*) malloc(leny * sizeof(float));
    float* ret = (float*) malloc((nsteps / save_skip + 1) * leny * sizeof(float));
    float* source, * dest; /* temporaries */
    int i, j;

    /* start with ret = y0 */
    memcpy(ret, y0, leny * sizeof(float));

    /* at each step, get ynew into ret, else edit in place */
    for (i = 0; i < nsteps / save_skip; i++) 
    {
        source = ret + i * leny;
        dest = source + leny; /* save into here */

        for (j = 0; j < save_skip; j++) 
        {
            step(f, dt, source, dest, dy, leny,
                    scratch, scratch2);
            
            /* run only if passed non-null */
            if(g != NULL)
            {
                (*g)(dest, leny);
            }
            /* first iteration of this loop doesn't overwrite source, next all do */
            source = dest;
        }
    }

    /* cleanup */
    free(dy);
    free(scratch);
    free(scratch2);
    return ret;
}
