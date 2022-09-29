#include "kernels.hpp"
#include <cassert>
#include <cstdlib>

struct gid {
  unsigned long gid;
  unsigned idx;
};

static int cmp_dofs(const void *a, const void *b) {
  struct gid *ga = (struct gid *)a;
  struct gid *gb = (struct gid *)b;

  if (ga->gid > gb->gid)
    return 1;
  else if (ga->gid < gb->gid)
    return -1;

  // ga->gid == gb->gid, so we look at idx
  if (ga->idx > gb->idx)
    return 1;
  else
    return -1;
}

int gs_setup(unsigned *gs_n, unsigned **gs_off_, unsigned **gs_ids_,
             unsigned nelt, unsigned nx1, unsigned ndim, unsigned verbose) {
  if (ndim != 3) {
    fprintf(stderr, "Only ndim = 3 is supported as of now.");
    exit(1);
  }
  if (nx1 < 2) {
    fprintf(stderr, "nx1 has to be greater than or equal to 2.");
    exit(1);
  }

  int nelx = cbrt(nelt + 1.0) + 1;
  while (nelx > 0 && nelt % nelx != 0)
    nelx--;
  unsigned nelt_ = nelt / nelx;

  int nely = sqrt(nelt_ + 1.0) + 1;
  while (nely > 0 && nelt_ % nely != 0)
    nely--;

  int nelz = nelt_ / nely;

  if (verbose > 0)
    printf("nelx = %d, nely = %d, nelz = %d\n", nelx, nely, nelz);

  unsigned ndofs = nelt * nx1 * nx1 * nx1;
  unsigned long *gids = (unsigned long *)calloc(ndofs, sizeof(unsigned long));

  unsigned i, j, k, nn = nx1 - 1;
  for (unsigned e = 0, dof = 0; e < nelt; e++) {
    // e = eg since we use only use one processor for now
    unsigned ex = e % nelx;
    unsigned ez = e / (nelx * nely);
    unsigned ey = (e - (ex + nelx * nely * ez)) / nelx;

    // Number dofs now
    for (i = 0; i < nx1; i++) {
      for (j = 0; j < nx1; j++) {
        for (k = 0; k < nx1; k++) {
          unsigned xd = nn * ex + i;
          unsigned yd = nn * ey + j;
          unsigned zd = nn * ez + k;
          gids[dof++] = xd + (nn * nelx + 1) * yd +
                        (nn * nelx + 1) * (nn * nely + 1) * zd + 1;
        }
      }
    }
  }

  if (verbose > 1) {
    for (unsigned i = 0; i < ndofs; i++)
      printf("gids[%d] = %lu\n", i, gids[i]);
  }

  struct gid *dofs = (struct gid *)calloc(ndofs, sizeof(struct gid));
  for (unsigned dof = 0; dof < ndofs; dof++) {
    dofs[dof].gid = gids[dof];
    dofs[dof].idx = dof;
  }

  qsort(dofs, ndofs, sizeof(struct gid), cmp_dofs);

  unsigned udofs = 0, gsdofs = 0;
  for (unsigned dof = 1, prev = 0; dof < ndofs; dof++) {
    if (dofs[dof].gid != dofs[prev].gid) {
      // only count global ids which are repeated
      if ((dof - prev) > 1) {
        udofs++;
        gsdofs += (dof - prev);
      }
      prev = dof;
    }
  }

  *gs_n = udofs;
  unsigned *gs_off = *gs_off_ = (unsigned *)calloc(udofs + 1, sizeof(unsigned));
  unsigned *gs_ids = *gs_ids_ = (unsigned *)calloc(gsdofs, sizeof(unsigned));

  udofs = 0;
  gs_off[0] = 0;
  for (unsigned dof = 1, prev = 0; dof < ndofs; dof++) {
    if (dofs[dof].gid != dofs[prev].gid) {
      if ((dof - prev) > 1) {
        for (e = prev; e < dof; e++)
          gs_ids[gs_off[udofs] + e - prev] = dofs[e].idx;
        udofs++;
        gs_off[udofs] = gs_off[udofs - 1] + dof - prev;
      }
      prev = dof;
    }
  }
  // Sanity check
  assert(gs_off[udofs] == gsdofs);

  free(gids), free(dofs);

  return 0;
}

int gs(scalar *u, unsigned n, unsigned *off, unsigned *id) {
  for (unsigned i = 0; i < n; i++) {
    scalar s = 0.0;
    for (unsigned j = off[i]; j < off[i + 1]; j++)
      s += u[id[j]];
    for (unsigned j = off[i]; j < off[i + 1]; j++)
      u[id[j]] = s;
  }

  return 0;
}
