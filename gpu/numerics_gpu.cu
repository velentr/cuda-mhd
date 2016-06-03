/* Author: Yubo Su
 * Numerics suite for cuda-mhd CPU implementation
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "numerics_gpu.h"

__global__ void dydx_exp(double *yn, double *dy, int leny)
{
    /* leny = NUM_COMPS * n^3, side length */
    int n = (int)cbrt((((double)leny) / NUM_COMPS));
    int i, j, k;
    int idx;
    /* store derivative indicies, left/right (store handling of boundary cases)
     * */
    int dxleft, dxright, dyleft, dyright, dzleft, dzright;
    double div_v; /* useul for precomputation */

    for (idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n * n * n;
         idx += blockDim.x * gridDim.x)
    {
        i = idx / (n * n);
        j = (idx % (n * n)) / n;
        k = idx % n;

        /* compute x derivative indicies */
        if (i == 0)
            dxleft = i;
        else
            dxleft = i - 1;
        if (i == n - 1)
            dxright = i;
        else
            dxright = i + 1;

        /* y derivative indicies */
        if (j == 0)
            dyleft = j;
        else
            dyleft = j - 1;
        if (j == n - 1)
            dyright = j;
        else
            dyright = j + 1;

        /* y derivative indicies */
        if (k == 0)
            dzleft = k;
        else
            dzleft = k - 1;
        if (k == n - 1)
            dzright = k;
        else
            dzright = k + 1;

        /* update terms */
        div_v =
            (yn[U2(n, dxright, j, k)] -
                yn[U2(n, dxleft, j, k)]) / ((dxright - dxleft) * DL) +
            (yn[U3(n, i, dyright, k)] -
                yn[U3(n, i, dyleft, k)]) / ((dyright - dyleft) * DL) +
            (yn[U4(n, i, j, dzright)] -
                yn[U4(n, i, j, dzleft)]) / ((dzright - dzleft) * DL);

        dy[U1(n, i, j, k)] = -div_v;

        dy[U2(n, i, j, k)] = -
            (yn[U2(n, i, j, k)] / yn[U1(n, i, j, k)]) *
                ((yn[U2(n, dxright, j, k)] -
                yn[U2(n, dxleft, j, k)]) / ((dxright - dxleft) * DL)) -
            (yn[U3(n, i, j, k)] / yn[U1(n, i, j, k)]) *
                ((yn[U2(n, i, dyright, k)] -
                yn[U2(n, i, dyleft, k)]) / ((dyright - dyleft) * DL)) -
            (yn[U4(n, i, j, k)] / yn[U1(n, i, j, k)]) *
                ((yn[U2(n, i, j, dzright)] -
                yn[U2(n, i, j, dzleft)]) / ((dzright - dzleft) * DL)) +

            yn[U5(n, i, j, k)] *
                ((yn[U2(n, dxright, j, k)] -
                yn[U2(n, dxleft, j, k)]) / ((dxright - dxleft) * DL)) +
            yn[U6(n, i, j, k)] *
                ((yn[U2(n, i, dyright, k)] -
                yn[U2(n, i, dyleft, k)]) / ((dyright - dyleft) * DL)) +
            yn[U7(n, i, j, k)] *
                ((yn[U2(n, i, j, dzright)] -
                yn[U2(n, i, j, dzleft)]) / ((dzright - dzleft) * DL)) -

            ((yn[U8(n, dxright, j, k)] -
                yn[U8(n, dxleft, j, k)]) / ((dxright - dxleft) * DL)) -

            (yn[U2(n, i, j, k)] / yn[U1(n, i, j, k)]) * div_v -

            (yn[U9(n, dxright, j, k)] -
                yn[U9(n, dxleft, j, k)]) / ((dxright - dxleft) * DL);

        dy[U3(n, i, j, k)] = -
            (yn[U2(n, i, j, k)] / yn[U1(n, i, j, k)]) *
                ((yn[U3(n, dxright, j, k)] -
                yn[U3(n, dxleft, j, k)]) / ((dxright - dxleft) * DL)) -
            (yn[U3(n, i, j, k)] / yn[U1(n, i, j, k)]) *
                ((yn[U3(n, i, dyright, k)] -
                yn[U3(n, i, dyleft, k)]) / ((dyright - dyleft) * DL)) -
            (yn[U4(n, i, j, k)] / yn[U1(n, i, j, k)]) *
                ((yn[U3(n, i, j, dzright)] -
                yn[U3(n, i, j, dzleft)]) / ((dzright - dzleft) * DL)) +

            yn[U5(n, i, j, k)] *
                ((yn[U3(n, dxright, j, k)] -
                yn[U3(n, dxleft, j, k)]) / ((dxright - dxleft) * DL)) +
            yn[U6(n, i, j, k)] *
                ((yn[U3(n, i, dyright, k)] -
                yn[U3(n, i, dyleft, k)]) / ((dyright - dyleft) * DL)) +
            yn[U7(n, i, j, k)] *
                ((yn[U3(n, i, j, dzright)] -
                yn[U3(n, i, j, dzleft)]) / ((dzright - dzleft) * DL)) -

            ((yn[U8(n, i, dyright, k)] -
                yn[U8(n, i, dyleft, k)]) / ((dyright - dyleft) * DL)) -

            (yn[U3(n, i, j, k)] / yn[U1(n, i, j, k)]) * div_v -

            (yn[U9(n, i, dyright, k)] -
                yn[U9(n, i, dyleft, k)]) / ((dyright - dyleft) * DL);

        dy[U4(n, i, j, k)] = -
            (yn[U2(n, i, j, k)] / yn[U1(n, i, j, k)]) *
                ((yn[U4(n, dxright, j, k)] -
                yn[U4(n, dxleft, j, k)]) / ((dxright - dxleft) * DL)) -
            (yn[U3(n, i, j, k)] / yn[U1(n, i, j, k)]) *
                ((yn[U4(n, i, dyright, k)] -
                yn[U4(n, i, dyleft, k)]) / ((dyright - dyleft) * DL)) -
            (yn[U4(n, i, j, k)] / yn[U1(n, i, j, k)]) *
                ((yn[U4(n, i, j, dzright)] -
                yn[U4(n, i, j, dzleft)]) / ((dzright - dzleft) * DL)) +

            yn[U5(n, i, j, k)] *
                ((yn[U4(n, dxright, j, k)] -
                yn[U4(n, dxleft, j, k)]) / ((dxright - dxleft) * DL)) +
            yn[U6(n, i, j, k)] *
                ((yn[U4(n, i, dyright, k)] -
                yn[U4(n, i, dyleft, k)]) / ((dyright - dyleft) * DL)) +
            yn[U7(n, i, j, k)] *
                ((yn[U4(n, i, j, dzright)] -
                yn[U4(n, i, j, dzleft)]) / ((dzright - dzleft) * DL)) -

            ((yn[U8(n, i, j, dzright)] -
                yn[U8(n, i, j, dzleft)]) / ((dzright - dzleft) * DL)) -

            (yn[U4(n, i, j, k)] / yn[U1(n, i, j, k)]) * div_v -

            (yn[U9(n, i, j, dzright)] -
                yn[U9(n, i, j, dzleft)]) / ((dzright - dzleft) * DL);

        dy[U5(n, i, j, k)] = (
            yn[U5(n, i, j, k)] *
                ((yn[U2(n, dxright, j, k)] -
                yn[U2(n, dxleft, j, k)]) / ((dxright - dxleft) * DL)) +
            yn[U6(n, i, j, k)] *
                ((yn[U2(n, i, dyright, k)] -
                yn[U2(n, i, dyleft, k)]) / ((dyright - dyleft) * DL)) +
            yn[U7(n, i, j, k)] *
                ((yn[U2(n, i, j, dzright)] -
                yn[U2(n, i, j, dzleft)]) / ((dzright - dzleft) * DL)) -

            yn[U2(n, i, j, k)] *
                ((yn[U5(n, dxright, j, k)] -
                yn[U5(n, dxleft, j, k)]) / ((dxright - dxleft) * DL)) -
            yn[U3(n, i, j, k)] *
                ((yn[U5(n, i, dyright, k)] -
                yn[U5(n, i, dyleft, k)]) / ((dyright - dyleft) * DL)) -
            yn[U4(n, i, j, k)] *
                ((yn[U5(n, i, j, dzright)] -
                yn[U5(n, i, j, dzleft)]) / ((dzright - dzleft) * DL)) -

            yn[U5(n, i, j, k)] * div_v) / yn[U1(n, i, j, k)];

        dy[U6(n, i, j, k)] = (
            yn[U5(n, i, j, k)] *
                ((yn[U3(n, dxright, j, k)] -
                yn[U3(n, dxleft, j, k)]) / ((dxright - dxleft) * DL)) +
            yn[U6(n, i, j, k)] *
                ((yn[U3(n, i, dyright, k)] -
                yn[U3(n, i, dyleft, k)]) / ((dyright - dyleft) * DL)) +
            yn[U7(n, i, j, k)] *
                ((yn[U3(n, i, j, dzright)] -
                yn[U3(n, i, j, dzleft)]) / ((dzright - dzleft) * DL)) -

            yn[U2(n, i, j, k)] *
                ((yn[U6(n, dxright, j, k)] -
                yn[U6(n, dxleft, j, k)]) / ((dxright - dxleft) * DL)) -
            yn[U3(n, i, j, k)] *
                ((yn[U6(n, i, dyright, k)] -
                yn[U6(n, i, dyleft, k)]) / ((dyright - dyleft) * DL)) -
            yn[U4(n, i, j, k)] *
                ((yn[U6(n, i, j, dzright)] -
                yn[U6(n, i, j, dzleft)]) / ((dzright - dzleft) * DL)) -

            yn[U6(n, i, j, k)] * div_v) / yn[U1(n, i, j, k)];

        dy[U7(n, i, j, k)] = (
            yn[U5(n, i, j, k)] *
                ((yn[U4(n, dxright, j, k)] -
                yn[U4(n, dxleft, j, k)]) / ((dxright - dxleft) * DL)) +
            yn[U6(n, i, j, k)] *
                ((yn[U4(n, i, dyright, k)] -
                yn[U4(n, i, dyleft, k)]) / ((dyright - dyleft) * DL)) +
            yn[U7(n, i, j, k)] *
                ((yn[U4(n, i, j, dzright)] -
                yn[U4(n, i, j, dzleft)]) / ((dzright - dzleft) * DL)) -

            yn[U2(n, i, j, k)] *
                ((yn[U7(n, dxright, j, k)] -
                yn[U7(n, dxleft, j, k)]) / ((dxright - dxleft) * DL)) -
            yn[U3(n, i, j, k)] *
                ((yn[U7(n, i, dyright, k)] -
                yn[U7(n, i, dyleft, k)]) / ((dyright - dyleft) * DL)) -
            yn[U4(n, i, j, k)] *
                ((yn[U7(n, i, j, dzright)] -
                yn[U7(n, i, j, dzleft)]) / ((dzright - dzleft) * DL)) -

            yn[U7(n, i, j, k)] * div_v) / yn[U1(n, i, j, k)];

        dy[U8(n, i, j, k)] =
            (yn[U10(n, dxright, j, k)] -
                yn[U10(n, dxleft, j, k)]) / ((dxright - dxleft) * DL) +
            (yn[U11(n, i, dyright, k)] -
                yn[U11(n, i, dyleft, k)]) / ((dyright - dyleft) * DL) +
            (yn[U12(n, i, j, dzright)] -
                yn[U12(n, i, j, dzleft)]) / ((dzright - dzleft) * DL);

        dy[U9(n, i, j, k)] = dy[U10(n, i, j, k)] = dy[U11(n, i, j, k)] =
            dy[U12(n, i, j, k)] = 0;
    }
}

__global__ void cuda_update(double *yn, int leny)
{
    /* Update computations of P^*, EPmDV */
    int n = (int)cbrt(((double)leny) / NUM_COMPS);
    int i, j, k;
    int idx;
    double bdotv; /* useful for precomputation */

    for (idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n * n * n;
         idx += blockDim.x * gridDim.x)
    {
        i = idx / (n * n);
        j = (idx % (n * n)) / n;
        k = idx % n;

        assert(i < n);
        assert(j < n);
        assert(k < n);

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


__global__ void cuda_step(double dt, double *yn, double *ynew, double *dy,
        int leny, double *scratch, double *scratch2, int run)
/* Given some dy/dt = f(y), compute the dy for a given yn and store into yn +
 * dy into ynew. Uses Runge-Kutta:
 * k1 = f(y), k2 = f(y + k1 * dt/2), k3 = f(y + k2 * dt/2),
 *      k4 = f(y + k3 * dt), ynew = yn + (k1 + 2k2 + 2k3 + k4)/6 * dt
 *
 * 2 scratch vectors required to not nuke in case ynew = yn (overwrite)
 *
 * Input:
 *  double dt            : timestep dt
 *  double* yn           : current y
 *  double* ynew         : new y
 *  double* dy           : get return from dydx (malloc in caller)
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

    switch (run)
    {
    case 0:
        /* scratch = yn + k1 * dt/2 */
        for (i = blockIdx.x * blockDim.x + threadIdx.x;
             i < leny;
             i += blockDim.x * gridDim.x)
        {
            scratch[i] = yn[i] + dy[i] * dt / 2; /* y + k1 * dt/2 */
            scratch2[i] = dy[i] / 6; /* k1 / 6 */
        }
        break;
    case 1:
        for (i = blockIdx.x * blockDim.x + threadIdx.x;
             i < leny;
             i += blockDim.x * gridDim.x)
        {
            scratch[i] = yn[i] + dy[i] * dt / 2; /* y + k2 * dt/2 */
            scratch2[i] += dy[i] / 3; /* (k1 + 2k2) / 6 */
        }
        break;
    case 2:
        for (i = blockIdx.x * blockDim.x + threadIdx.x;
             i < leny;
             i += blockDim.x * gridDim.x)
        {
            scratch[i] = yn[i] + dy[i] * dt; /* y + k3 * dt */
            scratch2[i] += dy[i] / 3; /* (k1 + 2k2 + 2k3) / 6 */
        }
        break;
    case 3:
        for (i = blockIdx.x * blockDim.x + threadIdx.x;
             i < leny;
             i += blockDim.x * gridDim.x)
        {
            /* dt * (k1 + 2k2 + 2k3 + k4) / 6 + yn */
            ynew[i] = dt * (scratch2[i] + dy[i] / 6) + yn[i];
        }
        break;
    default:
        assert(0);
    }
}

double *solve(double *y0, double dt, int nsteps, int leny, int save_skip,
        size_t nblk, size_t thr)
/* Given some initial y(t = 0) and dy/dt = f(y) (autonomous ODE), computes
 * y(nsteps * dt) by iterating y_n -> y_{n+1} via step. y assumed in
 * generality to be a vector
 *
 * Input:
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

    /* Run an initial update to get the correct P* and temp values. */
    cuda_update<<<nblk, thr>>>(y0, leny);

    /* start with ret = y0 */
    cudaMemcpy(ret_dev, y0, leny * sizeof(double), cudaMemcpyDeviceToDevice);

    /* at each step, get ynew into ret, else edit in place */
    for (i = 0; i < nsteps / save_skip; i++)
    {
        source = ret_dev + i * leny;
        dest = source + leny; /* save into here */

        for (j = 0; j < save_skip; j++)
        {
            /* Execute four steps in the simulation. */
            dydx_exp<<<nblk, thr>>>(source, dy, leny); /* compute k1 into dy */
            cuda_step<<<nblk, thr>>>(dt, source, dest, dy, leny, scratch,
                    scratch2, 0);
            dydx_exp<<<nblk, thr>>>(scratch, dy, leny); /* compute k2 into dy */
            cuda_step<<<nblk, thr>>>(dt, source, dest, dy, leny, scratch,
                    scratch2, 1);
            dydx_exp<<<nblk, thr>>>(scratch, dy, leny); /* compute k3 into dy */
            cuda_step<<<nblk, thr>>>(dt, source, dest, dy, leny, scratch,
                    scratch2, 2);
            dydx_exp<<<nblk, thr>>>(scratch, dy, leny); /* compute k4 into dy */
            cuda_step<<<nblk, thr>>>(dt, source, dest, dy, leny, scratch,
                    scratch2, 3);

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
    cudaFree(ret_dev);

    return ret;
}

