# LBM Sample
使用lattice Boltzmann method（LBM）计算顶盖驱动流（Lid driven flow）的 C++/CUDA 案例程序

GPU 并行加速有两种实现方式，分别是 NVIDIA CUDA 和 OpenACC

#### 参考文献

- Standard LBM 理论：Chen S., Doolen G.D. Lattice Boltzmann method for fluid flows [J]. Annual Review of Fluid Mechanics, 1998, 30: 329-364. https://doi:10.4249/scholarpedia.9507.
- GPU 编程：
  1. Tölke J., Krafczyk M. TeraFLOP computing on a desktop PC with GPUs for 3D CFD [J]. International Journal of Computational Fluid Dynamics, 2008: 22(7), 443-456. https://doi:10.1080/10618560802238275.
  2. CUDA C++ Programming Guide [1. Introduction — CUDA C Programming Guide (nvidia.com)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

- Swap 节省内存技术：Keijo M., Jari H., Tuomo R. et al. An efficient swap algorithm for the lattice Boltzmann method [J]. Computer Physics Communications, 2007, 176: 200-210. https://doi.org/10.1016/j.cpc.2006.09.005
- CPU 源代码：《格子Boltzmann方法的理论及应用》何雅玲
- MRT理论：《格子Boltzmann方法的原理及应用》郭照立，郑楚光
