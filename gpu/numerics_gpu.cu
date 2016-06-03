/* Author: Yubo Su
 * Numerics suite for cuda-mhd CPU implementation
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "numerics_gpu.h"

__global__ void cuda_update(double* yn, int leny)
{
    /* Update computations of P^*, EPmDV */
    int n = (int)cbrt((double)(leny / NUM_COMPS));
    int i, j, k;
    float bdotv; /* useul for precomputation */
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
        {
            for(k = 0; k < n; k++)
            {
                /* store this since used thrice */
                bdotv = yn[U2(n, i, j, k)] * yn[U5(n, i, j, k)] +
                    yn[U3(n, i, j, k)] * yn[U6(n, i, j, k)] +
                    yn[U4(n, i, j, k)] * yn[U7(n, i, j, k)];

                /* compute P^*, EPmDV */
                yn[U9(n, i, j, k)] = (GAMMA - 1) *
                    (yn[U8(n, i, j, k)] -
                        (yn[U2(n, i, j, k)] * yn[U2(n, i, j, k)] +
                         yn[U3(n, i, j, k)] * yn[U3(n, i, j, k)] +
                         yn[U4(n, i, j, k)] * yn[U4(n, i, j, k)]) /
                        (2 * yn[U1(n, i, j, k)]) +
                        (yn[U5(n, i, j, k)] * yn[U5(n, i, j, k)] +
                         yn[U6(n, i, j, k)] * yn[U6(n, i, j, k)] +
                         yn[U7(n, i, j, k)] * yn[U7(n, i, j, k)]) / 2);

                yn[U10(n, i, j, k)] = - (yn[U8(n, i, j, k)] + yn[U9(n, i, j, k)])
                    * yn[U2(n, i, j, k)] + yn[U5(n, i, j, k)] * bdotv;
                yn[U11(n, i, j, k)] = - (yn[U8(n, i, j, k)] + yn[U9(n, i, j, k)])
                    * yn[U3(n, i, j, k)] + yn[U6(n, i, j, k)] * bdotv;
                yn[U12(n, i, j, k)] = - (yn[U8(n, i, j, k)] + yn[U9(n, i, j, k)])
                    * yn[U4(n, i, j, k)] + yn[U7(n, i, j, k)] * bdotv;
            }
        }
    }
}


__global__ void cuda_step(void (*f)(double *, double *, int), double dt,
        double *yn, double *ynew, double *dy, int leny, double *scratch,
        double *scratch2)
/* Given some dy/dt = f(y), compute the dy for a given yn and store into yn +
 * dy into ynew. Uses Runge-Kutta:
 * k1 = f(y), k2 = f(y + k1 * dt/2), k3 = f(y + k2 * dt/2),
 *      k4 = f(y + k3 * dt), ynew = yn + (k1 + 2k2 + 2k3 + k4)/6 * dt
 *
 * 2 scratch vectors required to not nuke in case ynew = yn (overwrite)
 *
 * Input:
 *  void (*f)           : computes dy/dt = f(y), stores into second double*
 *  double dt            : timestep dt
 *  double* yn           : current y
 *  double* ynew         : new y
 *  double* dy           : get return from (*f) (malloc in caller)
 *  int leny            : length of y vector
 *  double* scratch[2]   : scratch vector, simply to avoid re-mallocing,
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

double* solve(void (*f)(double*, double*, int),
        double* y0, double dt, int nsteps, int leny, int save_skip,
        size_t nblk, size_t thr)
/* Given some initial y(t = 0) and dy/dt = f(y) (autonomous ODE), computes
 * y(nsteps * dt) by iterating y_n -> y_{n+1} via step. y assumed in
 * generality to be a vector
 *
 * Input:
 *  void (*f)           : computes dy/dt = f(y), stores into second double*
 *  double* init         : y(t=0)
 *  double dt            : dt timestep
 *  int nsteps          : number of timesteps to evolve
 *  int leny            : length of y vector
 *  int save_skip       : every save_skip timesteps, save trajectory
                        : use only divisible by nsteps please
 * Output:
 *  double*              : (nsteps + 1) * leny vector containing y(t), contains
 *                        timesteps dt * [0,nsteps / saveskip]
 */
{
    /* stores total trajectory */
    double *dy;
    double *scratch;
    double *scratch2;
    double *ret_dev;
    double *ret = (double *) malloc((nsteps / save_skip + 1) * leny * sizeof(double));
    double *source, * dest; /* temporaries */
    int i, j;

    /* Allocate memory for working set. */
    cudaMalloc((void **) &dy,        leny * sizeof(double));
    cudaMalloc((void **) &scratch,   leny * sizeof(double));
    cudaMalloc((void **) &scratch2,  leny * sizeof(double));

    /* Memory for output buffer. */
    cudaMalloc((void **)&ret_dev,
            (nsteps / save_skip + 1) * leny * sizeof(double));

    /* start with ret = y0 */
    cudaMemcpy(ret_dev, y0, leny * sizeof(double), cudaMemcpyDeviceToDevice);

    /* Run an initial update to get the correct P* and temp values. */
    cuda_update<<<nblk, thr>>>(y0, leny);

    /* at each step, get ynew into ret, else edit in place */
    for (i = 0; i < nsteps / save_skip; i++)
    {
        source = ret_dev + i * leny;
        dest = source + leny; /* save into here */

        for (j = 0; j < save_skip; j++)
        {
            /* Execute a single step in the simulation. */
            cuda_step<<<nblk, thr>>>(f, dt, source, dest, dy, leny, scratch,
                    scratch2);

            /* Update the dataset. */
            cuda_update<<<nblk, thr>>>(dest, leny);

            /* first iteration of this loop doesn't overwrite source */
            source = dest;
        }
    }

    /* Copy return buffer back to host. */
    cudaMemcpy(ret, ret_dev, (nsteps / save_skip + 1) * leny * sizeof(double),
            cudaMemcpyDeviceToHost);

    /* cleanup */
    cudaFree(dy);
    cudaFree(scratch);
    cudaFree(scratch2);

    return ret;
}
