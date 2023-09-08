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

double w[Q] = { 4.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 36,1.0 / 36,1.0 / 36,1.0 / 36 }; //Ȩϵ��
double rho[NX][NY], u[NX][NY][2], u0[NX][NY][2], f[NX][NY][Q], F[NX][NY][Q];
//�ܶȣ�n+1ʱ���ٶȣ�nʱ���ٶȣ��ݻ�ǰ�ܶȷֲ��������ݻ����ܶȷֲ�����
//int i, j, k, ip, jp, n; //k��Ϊalpha
double error;

void init(); //��ʼ������
double feq(int k, double rho, double u[2]); //ƽ��̬����
void evolution();
//�ݻ�����
void output(int m); //�������
void Error(); //����

#endif