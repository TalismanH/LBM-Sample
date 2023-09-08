//#pragma once
#ifndef BASICVAR_H_ //�ļ���ͷ
	#define BASICVAR_H_ 

using namespace std; //�����ռ�

const int Q = 9; //D2Q9ģ��
const int e[Q][2] = { {0,0},{1,0},{0,1},{-1,0},{0,-1},{1,1},{-1,1},{-1,-1},{1,-1} }; //��ɢ�ٶ�����
const int NX = 128; //x���򣬽ڵ���NX+1,NY+1
const int NY = 128; //y����
const double U = 0.1; //�����ٶ�
const double rho0 = 1.0;
const int Re = 1000;
const double niu = U * NX / Re;
const double tau_f = 3.0 * niu + 0.5;

void init(); //��ʼ������
double feq(int k, double rho, double u[2]); //ƽ��̬����
void evolution(double rho[NX + 1][NY + 1], double u[NX + 1][NY + 1][2], double u0[NX + 1][NY + 1][2], double f[NX + 1][NY + 1][Q], double F[NX + 1][NY + 1][Q]);
//�ݻ�����
void output(int m); //�������
void Error(); //����

#endif