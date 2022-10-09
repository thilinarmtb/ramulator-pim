#include "kernels.hpp"
#include <cstdio>
#include <cstdlib>

#define I2(i, j) (i + nx1 * j)
#define I3(i, j, k) (i + nx1 * j + nx1 * nx1 * k)
#define I4(i, j, k, e) (I3(i, j, k) + nx1 * nx1 * nx1 * e)

void ax_v00(scalar *w, unsigned E, unsigned nx1, const scalar *u, unsigned ngeo,
            const scalar *geo, const scalar *D, scalar *wrk) {
  if (ngeo != 6) {
    fprintf(stderr, "Only ngeo = 6 supported !");
    exit(1);
  }

  unsigned nx3 = nx1 * nx1 * nx1;
  scalar *ur = wrk, *us = ur + nx3, *ut = us + nx3;
  unsigned dofs = E * nx3;

#if defined(OMP_OFFLOAD)
#pragma omp target teams num_teams(E)
#endif
#pragma omp parallel for
  for (unsigned e = 0; e < E; e++) {
    for (unsigned i = 0; i < nx1; i++) {
      for (unsigned j = 0; j < nx1; j++) {
        for (unsigned k = 0; k < nx1; k++) {
          scalar r_G00 = geo[0 * dofs + I3(i, j, k)];
          scalar r_G01 = geo[1 * dofs + I3(i, j, k)];
          scalar r_G02 = geo[2 * dofs + I3(i, j, k)];
          scalar r_G11 = geo[3 * dofs + I3(i, j, k)];
          scalar r_G12 = geo[4 * dofs + I3(i, j, k)];
          scalar r_G22 = geo[5 * dofs + I3(i, j, k)];

          scalar r_ur = 0.0, r_us = 0, r_ut = 0;
          for (unsigned m = 0; m < nx1; m++) {
            r_ur += D[I2(m, i)] * u[I4(m, j, k, e)];
            r_us += D[I2(m, j)] * u[I4(i, m, k, e)];
            r_ut += D[I2(m, k)] * u[I4(i, j, m, e)];
          }

          ur[I3(i, j, k)] = r_G00 * r_ur + r_G01 * r_us + r_G02 * r_ut;
          us[I3(i, j, k)] = r_G01 * r_ur + r_G11 * r_us + r_G12 * r_ut;
          ut[I3(i, j, k)] = r_G02 * r_ur + r_G12 * r_us + r_G22 * r_ut;
        }
      }
    }

    for (unsigned i = 0; i < nx1; i++) {
      for (unsigned j = 0; j < nx1; j++) {
        for (unsigned k = 0; k < nx1; k++) {
          scalar wr = 0.0;
          for (unsigned m = 0; m < nx1; m++) {
            wr += D[I2(i, m)] * ur[I3(m, j, k)];
            wr += D[I2(j, m)] * us[I3(i, m, k)];
            wr += D[I2(k, m)] * ut[I3(i, j, m)];
          }
          w[I4(i, j, k, e)] = wr;
        }
      }
    }
  }
}

#undef I4
#undef I3
#undef I2
