//�ļ���ͷ����ֹ�ظ�����ͷ�ļ�
#ifndef BASICVAR_H_ 
	#define BASICVAR_H_ 

using namespace std; //�����ռ�

/********************��������*************************/

const int e[9][2] = { {0,0},{1,0},{0,1},{-1,0},{0,-1},{1,1},{-1,1},{-1,-1},{1,-1} }; //��ɢ�ٶ�����
const double w[9] = { 4.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 36,1.0 / 36,1.0 / 36,1.0 / 36 }; //Ȩϵ��
const int NX = 128; //x����
const int NY = 128; //y����
const double U = 0.1; //�����ٶ�
const double rho0 = 1.0;
const int Re = 1000;
const double niu = U * NX / Re;
const double tau_f = 3.0 * niu + 0.5;

/***********************����ԭ��**********************/

void init(); //��ʼ������
double feq(const int k, const double rho, const double ux, const double uy); //ƽ��̬�ֲ�����
void evolution(double* rho, double* ux, double* uy, double* u0x, double* u0y,
			   double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
			   int* Flag);
//�ݻ�����
void output(int m); //�������
double Error(const double * ux, const double * uy, const double * u0x, const double * u0y); //����

void memoryInitiate(); //�����ڴ�
void memoryfinalize(); //����ڴ�

#endif