#include "kernels.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <omp.h>

#include "../../misc/hooks/zsim_hooks.h"

#define M 10

static inline void init_vec(scalar *v, unsigned size) {
  for (unsigned i = 0; i < size; i++)
    v[i] = (M * rand() + 0.5) / RAND_MAX;
}

void copy(scalar *a, scalar *b, unsigned n) {
  zsim_PIM_function_begin();
#if defined(OMP_OFFLOAD)
#pragma omp target teams
#endif
#pragma omp parallel for
  for (unsigned i = 0; i < n; i++)
    a[i] = b[i];
  zsim_PIM_function_end();
}

void add2s1(scalar *a, scalar *b, scalar c, unsigned n) {
  zsim_PIM_function_begin();
#if defined(OMP_OFFLOAD)
#pragma omp target teams
#endif
#pragma omp parallel for
  for (unsigned i = 0; i < n; i++)
    a[i] = c * a[i] + b[i];
  zsim_PIM_function_end();
}

void add2s2(scalar *a, scalar *b, scalar c, unsigned n) {
  zsim_PIM_function_begin();
#if defined(OMP_OFFLOAD)
#pragma omp target teams
#endif
#pragma omp parallel for
  for (unsigned i = 0; i < n; i++)
    a[i] = a[i] + c * b[i];
  zsim_PIM_function_end();
}

void glsc3(scalar *a, scalar *b, scalar *c, unsigned n) {
  zsim_PIM_function_begin();
  scalar sum = 0;
#if defined(OMP_OFFLOAD)
#pragma omp target teams
#endif
#pragma omp parallel for reduction(+ : sum)
  for (unsigned i = 0; i < n; i++)
    sum += a[i] * b[i] * c[i];
  zsim_PIM_function_end();
}

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr,
            "Usage: %s <num_threads> <number_of_elements> <polynomial_order> "
            "<kernel_id>\n",
            argv[0]);
    exit(1);
  }

  // Set number of threads
  unsigned num_threads = atoi(argv[1]);
  omp_set_num_threads(num_threads);

  // Read in problem size and determine the vector size
  unsigned E = atoi(argv[2]);
  unsigned p = atoi(argv[3]);
  unsigned nx1 = p + 1;
  unsigned dofs = E * nx1 * nx1 * nx1;

  // Fix the problem dimension to 3
  const int ndim = 3;
  const int ngeo = ((ndim + 1) * ndim) / 2;

  // Random constant alpha
  srand((unsigned)time(NULL));
  scalar alpha = (M * rand() + 0.5) / RAND_MAX;

  // Allocate and initialize the vectors: a, b, c
  scalar *a = (scalar *)calloc(dofs, sizeof(scalar));
  init_vec(a, dofs);
  scalar *b = (scalar *)calloc(dofs, sizeof(scalar));
  init_vec(b, dofs);
  scalar *c = (scalar *)calloc(dofs, sizeof(scalar));
  init_vec(c, dofs);

  // For ax we need a few more vectors: g, D
  scalar *g = (scalar *)calloc(ngeo * dofs, sizeof(scalar));
  init_vec(g, ngeo * dofs);

  unsigned nx2 = nx1 * nx1;
  scalar *D = (scalar *)calloc(nx2, sizeof(scalar));
  init_vec(D, nx2);

  // Work arrays for ax
  unsigned wrk_size = ndim * nx1 * nx2;
  scalar *wrk = (scalar *)calloc(wrk_size, sizeof(scalar));

  zsim_roi_begin();

#if defined(OMP_OFFLOAD)
  if (num_threads > 1) {
    fprintf(stderr,
            "OpenMP offload version shold be run with a single thread\n");
    exit(1);
  }
  omp_set_default_device(0);
#pragma omp target enter data map(to                                           \
                                  : a [0:dofs], b [0:dofs], c [0:dofs],        \
                                    g [0:6 * dofs], D [0:nx2],                 \
                                    wrk [0:wrk_size])
#endif

  struct gs_data *gsh = NULL;
  unsigned kernel = atoi(argv[4]);
  switch (kernel) {
  case 10:
    // Call copy kernel
    copy(a, b, dofs);
    break;
  case 20:
    add2s1(a, b, alpha, dofs);
    break;
  case 30:
    add2s2(a, b, alpha, dofs);
    break;
  case 40:
    glsc3(a, b, c, dofs);
    break;
  case 50:
    gsh = gs_setup(E, nx1, ndim, 1);
    gs(a, gsh);
    gs_free(gsh);
    break;
  case 60:
    ax_v00(b, E, nx1, a, ngeo, g, D, wrk);
  default:
    fprintf(stderr, "Invalid kernel id: %u\n", kernel);
    break;
  }

#if defined(OMP_OFFLOAD)
#pragma omp target exit data map(delete                                        \
                                 : a [0:dofs], b [0:dofs], c [0:dofs],         \
                                   g [0:6 * dofs], D [0:nx2],                  \
                                   wrk [0:wrk_size])
#endif

  zsim_roi_end();

  free(a), free(b), free(c), free(g), free(D), free(wrk);

  return 0;
}

#undef M
