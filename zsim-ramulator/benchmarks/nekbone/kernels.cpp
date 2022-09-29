#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <omp.h>

#include "../../misc/hooks/zsim_hooks.h"

#define M 10

void copy(double *a, double *b, unsigned n) {
#if defined(OMP_OFFLOAD)
#pragma omp target
  {
#endif

    zsim_PIM_function_begin();
#pragma omp parallel for
    for (unsigned i = 0; i < n; i++)
      a[i] = b[i];
    zsim_PIM_function_end();

#if defined(OMP_OFFLOAD)
  }
#endif
}

void add2s1(double *a, double *b, double c, unsigned n) {
#if defined(OMP_OFFLOAD)
#pragma omp target
  {
#endif

    zsim_PIM_function_begin();
#pragma omp parallel for
    for (unsigned i = 0; i < n; i++)
      a[i] = c * a[i] + b[i];
    zsim_PIM_function_end();

#if defined(OMP_OFFLOAD)
  }
#endif
}

void add2s2(double *a, double *b, double c, unsigned n) {
#if defined(OMP_OFFLOAD)
#pragma omp target
  {
#endif

    zsim_PIM_function_begin();
#pragma omp parallel for
    for (unsigned i = 0; i < n; i++)
      a[i] = a[i] + c * b[i];
    zsim_PIM_function_end();

#if defined(OMP_OFFLOAD)
  }
#endif
}

void glsc3(double *a, double *b, double *c, unsigned n) {
#if defined(OMP_OFFLOAD)
#pragma omp target
  {
#endif

    double sum = 0;
    zsim_PIM_function_begin();
#pragma omp parallel for reduction(+ : sum)
    for (unsigned i = 0; i < n; i++)
      sum += a[i] * b[i] * c[i];
    zsim_PIM_function_end();

#if defined(OMP_OFFLOAD)
  }
#endif
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

  // Allocate the vectors
  unsigned E = atoi(argv[2]);
  unsigned p = atoi(argv[3]);
  unsigned dofs = E * (p + 1) * (p + 1) * (p + 1);

  // Init random number generation
  srand((unsigned)time(NULL));
  double alpha = (double)(M * rand() + 0.5) / RAND_MAX;
  double *a = (double *)calloc(dofs, sizeof(double));
  double *b = (double *)calloc(dofs, sizeof(double));
  double *c = (double *)calloc(dofs, sizeof(double));

  for (unsigned i = 0; i < dofs; i++) {
    a[i] = (double)(M * rand() + 0.5) / RAND_MAX;
    b[i] = (double)(M * rand() + 0.5) / RAND_MAX;
    c[i] = (double)(M * rand() + 0.5) / RAND_MAX;
  }

  zsim_roi_begin();

#if defined(OMP_OFFLOAD)
  if (num_threads > 1) {
    fprintf(stderr,
            "OpenMP offload version shold be run with a single thread\n");
    exit(1);
  }
  omp_set_default_device(0);
#pragma omp target enter data map(to : a [0:dofs], b [0:dofs], c [0:dofs])
#endif

  unsigned kernel = atoi(argv[4]);
  switch (kernel) {
  case 0:
    // Call copy kernel
    copy(a, b, dofs);
    break;
  case 1:
    add2s1(a, b, alpha, dofs);
    break;
  case 2:
    add2s2(a, b, alpha, dofs);
    break;
  case 3:
    glsc3(a, b, c, dofs);
    break;
  default:
    fprintf(stderr, "Invalid kernel id: %u\n", kernel);
    break;
  }

#if defined(OMP_OFFLOAD)
#pragma omp target exit data map(delete : a [0:dofs], b [0:dofs], c [0:dofs])
#endif

  zsim_roi_end();

  free(a), free(b), free(c);

  return 0;
}
