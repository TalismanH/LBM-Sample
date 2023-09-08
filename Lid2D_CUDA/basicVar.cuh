//�ļ���ͷ����ֹ�ظ�����ͷ�ļ�
#ifndef BASICVAR_CUH_ 
#define BASICVAR_CUH_ 

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std; //�����ռ�

/********************��������*************************/

//const int e[9][2] = { {0,0},{1,0},{0,1},{-1,0},{0,-1},{1,1},{-1,1},{-1,-1},{1,-1} }; //��ɢ�ٶ�����
//const double w[9] = { 4.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 36,1.0 / 36,1.0 / 36,1.0 / 36 }; //Ȩϵ��
const int NX = 256; //x����
const int NY = 256; //y����
const double U = 0.1; //�����ٶ�
const double rho0 = 1.0;
const int Re = 1000;
const double niu = U * NX / Re;
const double tau_f = 3.0 * niu + 0.5;

/***********************����ԭ��**********************/

void initializer(); //��ʼ������
//double feq(const int k, const double rho, const double ux, const double uy); //ƽ��̬�ֲ�����
//void evolution(double* rho, double* ux, double* uy,
//	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8);
//�ݻ�����
void output(int m); //�������
double Error(const double* ux, const double* uy, const double* u0x, const double* u0y); //����

void memoryInitiate(); //�����ڴ�
void memoryfinalize(); //����ڴ�

inline cudaError_t checkCuda(cudaError_t result);

//1. ����ƽ��̬�ֲ�����
__device__ double feq(const int k, const double rho, const double ux, const double uy);

__global__ void LBInitializer(const int NX, const int NY, const double rho0, const double U, double* ux, double* uy, double* u0x, double* u0y, double* rho, int* Flag,
	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8);

//2. ��ײ
__global__ void LBColl(double* rho, double* ux, double* uy, double* f0, double* f1, double* f2, double* f3,
	double* f4, double* f5, double* f6, double* f7, double* f8, int* Flag, const double tau_f);

__global__ void LBBoundary(const int NX, const int NY, double* rho, double* ux, double* uy, double* f0, double* f1, double* f2, double* f3,
	double* f4, double* f5, double* f6, double* f7, double* f8, int* Flag);

__global__ void LBProp(const int NX, const int NY, double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
	double* F0, double* F1, double* F2, double* F3, double* F4, double* F5, double* F6, double* F7, double* F8, int* Flag);

__global__ void LBUpgrade(double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
	double* F0, double* F1, double* F2, double* F3, double* F4, double* F5, double* F6, double* F7, double* F8, int* Flag);

__global__ void LBmacro_kernel(double* rho, double* ux, double* uy, double* u0x, double* u0y,
	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
	int* Flag);


#endif