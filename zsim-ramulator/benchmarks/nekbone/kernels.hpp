#if !defined(_RAMULATOR_PIM_KERNELS_HPP_)
#define _RAMULATOR_PIM_KERNELS_HPP_

#define scalar double

int gs_setup(unsigned *n, unsigned **off, unsigned **ids, unsigned nelt,
             unsigned nx1, unsigned ndim, unsigned verbose);
int gs(scalar *u, unsigned n, unsigned *off, unsigned *id);

#endif
