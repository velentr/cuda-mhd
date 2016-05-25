#include "numerics_cpu.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
/* MHD code!
 *
 * Step size of 1 is assumed
 */

#define NUM_COMPS 12
#define U1(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)))
#define U2(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 1)
#define U3(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 2)
#define U4(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 3)
#define U5(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 4)
#define U6(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 5)
#define U7(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 6)
#define U8(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 7)
#define U9(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 8)
#define U10(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 9)
#define U11(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 10)
#define U12(n, x, y, z) (NUM_COMPS * (((n) * (n) * (x)) + ((n) * (y)) + (z)) + 11)
#define MU 1
#define DL 1000
#define GAMMA 1.4

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

void update(double* yn, int leny)
{
    /* Update computations of P^*, EPmDV */
    int n = cbrt(leny / NUM_COMPS);
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

int main(int argc, const char *argv[])
{
    /* 
     * test code
     * */
    double *y0, *ret; /* initial condition, return */
    double dt, finalT;
    int leny, nsteps, n, save_skip;
    int i, j, k, l;

    /* run parameters parameters */
    n = 30; /* side length of cube in # grids */
    leny = NUM_COMPS * n * n * n;

    finalT = 0.000001;
    nsteps = 100;
    dt = finalT / nsteps;
    save_skip = nsteps / 2;

    y0 = (double*) malloc(leny * sizeof(double));

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
    update(y0, leny); /* get the right P^*, temp values */

    /* run solve() */
    ret = solve(dydx_exp, update, y0, dt, nsteps, leny, save_skip);
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
    return 0;
}
