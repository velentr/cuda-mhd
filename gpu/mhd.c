#include "numerics_gpu.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>

/* MHD code!
 *
 * Step size of 1 is assumed
 */

void dydx_exp(double* yn, double* dy, int leny)
{
    /* leny = NUM_COMPS * n^3, side length */
    int n = cbrt(leny / NUM_COMPS);
    int i, j, k;
    /* store derivative indicies, left/right (store handling of boundary cases)
     * */
    int dxleft, dxright, dyleft, dyright, dzleft, dzright;
    float div_v; /* useul for precomputation */

    /* change each point */
    for(i = 0; i < n; i++)
    {
        /* compute x derivative indicies */
        if (i == 0)
        {
            dxleft = i;
        }
        else
        {
            dxleft = i - 1;
        }
        if (i == n - 1)
        {
            dxright = i;
        }
        else
        {
            dxright = i + 1;
        }
        for(j = 0; j < n; j++)
        {
            /* y derivative indicies */
            if (j == 0)
            {
                dyleft = j;
            }
            else
            {
                dyleft = j - 1;
            }
            if (j == n - 1)
            {
                dyright = j;
            }
            else
            {
                dyright = j + 1;
            }
            for(k = 0; k < n; k++)
            {
                /* y derivative indicies */
                if (k == 0)
                {
                    dzleft = k;
                }
                else
                {
                    dzleft = k - 1;
                }
                if (k == n - 1)
                {
                    dzright = k;
                }
                else
                {
                    dzright = k + 1;
                }

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
    }
}

int main(int argc, const char *argv[])
{
    /*
     * test code
     * */
    double *y0, *y0_dev, *ret; /* initial condition, return */
    double dt, finalT;
    int leny, nsteps, n, save_skip;
    int i, j, k, l;
    int nblks, nthrds;

    if (argc != 3)
    {
        printf("usage: %s <nthrds> <nblks>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    nthrds = atoi(argv[1]);
    nblks  = atoi(argv[2]);

    /* run parameters parameters */
    n = 30; /* side length of cube in # grids */
    leny = NUM_COMPS * n * n * n;

    finalT = 0.000001;
    nsteps = 100;
    dt = finalT / nsteps;
    save_skip = nsteps / 2;

    /* Allocate data on the GPU. */
    y0 = (double*) malloc(leny * sizeof(*y0));
    cudaMalloc((void **)&y0_dev, leny * sizeof(*y0_dev));

    /* initial value */
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
        {
            for(k = 0; k < n; k++)
            {
                y0[U1(n, i, j, k)] = 1000; /* random large initial density */
                /* set each cell */
                for (l = 1; l < NUM_COMPS; l++)
                {
                    y0[U1(n, i, j, k) + l] = i * (n - i)
                        * j * (n - j) * k * (n - k) / (n * n * n) + 0.01;
                }
            }
        }
    }

    /* Copy initial data over to the GPU. */
    cudaMemcpy(y0_dev, y0, leny * sizeof(*y0), cudaMemcpyHostToDevice);

    /* run solve() */
    ret = solve(dydx_exp, y0_dev, dt, nsteps, leny, save_skip, nblks, nthrds);
    for(i = 0; i < nsteps / save_skip + 1; i++)
    {
        printf("[%f", ret[i * leny]);
        for (j = 1; j < leny; j++)
        {
            printf(", %f", ret[i * leny + j]);
        }
        printf("]\n");
    }

    /* cleanup */
    free(y0);
    free(ret);
    cudaFree(y0_dev);

    return 0;
}
