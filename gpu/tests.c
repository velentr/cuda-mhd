#include "numerics_gpu.h"
#include <stdio.h>
#include <math.h>
/* Test suite for numerics_cpu
 */

/* values for ODE testing */
#define C1 1
#define C2 (1.0 / 3)

void dydx_exp(double* yn, double* dy, int leny)
{
    /* computes dy_k/dx = (-1)^k*y_k */
    int i;
    for (i = 0; i < leny; i++)
    {
        dy[i] = (i % 2 == 0)? C1 * yn[i] : C2 * yn[i];
    }
}

int main(int argc, const char *argv[])
{
    /*
     * test code
     * */
    double *y0, *ret; /* initial condition, return */
    double dt, finalT;
    int leny, nsteps, save_skip;

    /* global parameters */
    leny = 1; /* y is length 1 */
    finalT = 10;
    y0 =  (double*) malloc(leny * sizeof(double));
    y0[0] = 1;
    save_skip = 1000;

    /* run 1 parameters */
    dt = 0.01;
    nsteps = (int) finalT / dt;
    /* run solve() */
    printf("Testing dy/dx = [%d * y], Runge-Kutta\n", C1);
    ret = solve(dydx_exp, NULL, y0, dt, nsteps, leny, save_skip);
    printf("Final T:%f\tdt:%f\tFinal Value:%f\n", finalT, dt, ret[nsteps / save_skip]);
    free(ret);

    /* run 2 parameters */
    dt = 0.001;
    nsteps = (int) finalT / dt;
    /* run solve() */
    ret = solve(dydx_exp, NULL, y0, dt, nsteps, leny, save_skip);
    printf("Final T:%f\tdt:%f\tFinal Value:%f\n", finalT, dt, ret[nsteps / save_skip]);
    free(ret);
    printf("True value:%f\n", exp(finalT));



    /* let's test vector-valued function */
    printf("\nNow testing multiple values!\n");
    leny = 2; /* y is length 1 */
    free(y0);
    y0 =  (double*) malloc(leny * sizeof(double));
    y0[0] = 1;
    y0[1] = 1;

    /* run 1 parameters */
    dt = 0.01;
    nsteps = (int) finalT / dt;
    /* run solve() */
    printf("Testing dy/dx = [%d * y, %f * y], Runge-Kutta\n", C1, C2);
    ret = solve(dydx_exp, NULL, y0, dt, nsteps, leny, save_skip);
    printf("Final T:%f\tdt:%f\tFinal Value:(%f, %f)\n",
            finalT, dt, ret[2 * (nsteps / save_skip)], ret[2 * (nsteps / save_skip) + 1]);
    free(ret);

    /* run 2 parameters */
    dt = 0.001;
    nsteps = (int) finalT / dt;
    /* run solve() */
    ret = solve(dydx_exp, NULL, y0, dt, nsteps, leny, save_skip);
    printf("Final T:%f\tdt:%f\tFinal Value:(%f, %f)\n",
            finalT, dt, ret[2 * (nsteps / save_skip)], ret[2 * (nsteps / save_skip) + 1]);
    free(ret);
    /*
     * End Forward Euler test code
     * */

    /* print true value */
    printf("True value:(%f, %f)\n", exp(C1 * finalT), exp(C2 * finalT));

    /* cleanup */
    free(y0);
    return 0;
}
