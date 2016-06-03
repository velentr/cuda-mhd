#include "numerics_gpu.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>

/* MHD code!
 *
 * Step size of 1 is assumed
 */

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
    n = 20; /* side length of cube in # grids */
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
    ret = solve(y0_dev, dt, nsteps, leny, save_skip, nblks, nthrds);
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
