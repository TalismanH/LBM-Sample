//文件开头，防止重复编译头文件
#ifndef BASICVAR_H_ 
	#define BASICVAR_H_ 

using namespace std; //命名空间

/********************常量声明*************************/

const int e[9][2] = { {0,0},{1,0},{0,1},{-1,0},{0,-1},{1,1},{-1,1},{-1,-1},{1,-1} }; //离散速度张量
const double w[9] = { 4.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 36,1.0 / 36,1.0 / 36,1.0 / 36 }; //权系数
const int NX = 128; //x方向
const int NY = 128; //y方向
const double U = 0.1; //顶盖速度
const double rho0 = 1.0;
const int Re = 1000;
const double niu = U * NX / Re;
const double tau_f = 3.0 * niu + 0.5;

/***********************函数原型**********************/

void init(); //初始化函数
double feq(const int k, const double rho, const double ux, const double uy); //平衡态分布函数
void evolution(double* rho, double* ux, double* uy, double* u0x, double* u0y,
			   double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
			   int* Flag);
//演化函数
void output(int m); //输出函数
double Error(const double * ux, const double * uy, const double * u0x, const double * u0y); //误差函数

void memoryInitiate(); //分配内存
void memoryfinalize(); //清除内存

#endif