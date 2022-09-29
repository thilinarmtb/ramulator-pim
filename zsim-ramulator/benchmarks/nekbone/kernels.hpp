#if !defined(_RAMULATOR_PIM_KERNELS_HPP_)
#define _RAMULATOR_PIM_KERNELS_HPP_

#define scalar double

struct gs_data;
struct gs_data *gs_setup(unsigned nelt, unsigned nx1, unsigned ndim,
                         unsigned verbose);
void gs(scalar *u, struct gs_data *gsd);
void gs_free(struct gs_data *gsd);

#endif
