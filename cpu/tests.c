#include "numerics_cpu.h"
#include <stdio.h>
#include <math.h>
/* Test suite for numerics_cpu 
 */

void dydx_exp(float* yn, float* dy, int leny)
{
    /* computes dy_k/dx = (-1)^k*y_k */
    int i;
    for (i = 0; i < leny; i++) 
    {
        dy[i] = (i % 2 == 0)? yn[i] : -yn[i];
    }
}

int main(int argc, const char *argv[])
{
    /* 
     * Forward Euler test code
     * */
    float *y0, *ret; /* initial condition, return */
    float dt, finalT;
    int leny, nsteps;

    /* global parameters */
    leny = 1; /* y is length 1 */
    finalT = 10;
    y0 =  (float*) malloc(leny * sizeof(float));
    *y0 = 1;

    /* run 1 parameters */
    dt = 0.01;
    nsteps = (int) finalT / dt;
    /* run solve() */
    printf("Testing dy/dx = y, Forward Euler\n");
    ret = solve(dydx_exp, y0, dt, nsteps, leny);
    printf("Final T:%f\tdt:%f\tFinal Value:%f\n", finalT, dt, ret[nsteps]);
    free(ret);

    /* run 2 parameters */
    dt = 0.001;
    nsteps = (int) finalT / dt;
    /* run solve() */
    ret = solve(dydx_exp, y0, dt, nsteps, leny);
    printf("Final T:%f\tdt:%f\tFinal Value:%f\n", finalT, dt, ret[nsteps]);
    free(ret);
    printf("True value:%f\n", exp(finalT));

    printf("\nNow testing multiple values!\n");
    /* run 3 parameters */
    leny = 2; /* y is length 1 */
    finalT = 10;
    free(y0);
    y0 =  (float*) malloc(leny * sizeof(float));
    y0[0] = 1;
    y0[1] = 1;

    /* run 1 parameters */
    dt = 0.01;
    nsteps = (int) finalT / dt;
    /* run solve() */
    printf("Testing dy/dx = y, Forward Euler\n");
    ret = solve(dydx_exp, y0, dt, nsteps, leny);
    printf("Final T:%f\tdt:%f\tFinal Value:(%f, %f)\n", 
            finalT, dt, ret[2 * nsteps], ret[2 * nsteps + 1]);
    free(ret);

    /* run 2 parameters */
    dt = 0.001;
    nsteps = (int) finalT / dt;
    /* run solve() */
    ret = solve(dydx_exp, y0, dt, nsteps, leny);
    printf("Final T:%f\tdt:%f\tFinal Value:(%f, %f)\n", 
            finalT, dt, ret[2 * nsteps], ret[2 * nsteps + 1]);
    free(ret);
    /* 
     * End Forward Euler test code
     * */

    /* print true value */
    printf("True value:(%f, %f)\n", exp(finalT), exp(-finalT));

    /* cleanup */
    free(y0);
    return 0;
}
