#if !defined(_RAMULATOR_PIM_KERNELS_HPP_)
#define _RAMULATOR_PIM_KERNELS_HPP_

#define scalar double

struct gs_data;
struct gs_data *gs_setup(unsigned nelt, unsigned nx1, unsigned ndim,
                         unsigned verbose);
int gs(scalar *u, unsigned n, unsigned *off, unsigned *id);

#endif
