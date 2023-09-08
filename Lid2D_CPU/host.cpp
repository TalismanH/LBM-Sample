
/*
CPUver4.0 �� ver3.0 �����ϼ����� Flag ��Ǻ� Swap

2021.2.27 Swap δʵ��
2021.2.27 22:30 Swap ��ʵ�֣��������ν�������һ�κ����ڽڵ㽻�����ڶ��νڵ��ڲ�������
				
ʹ�� Swap ǰ����ʱ��Ϊ250�����ң�ʹ�ú�Ϊ330������
δ�ﵽ���������ġ�������� 1.2 - 1.3 ����������Ľ�
*/

//����ͷ�ļ�
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>
#include "basicVar.h"

//��������
double* rho;												//�ܶ�
double* ux, * uy;											//�ٶ�
double* u0x, * u0y;											//������һʱ�����ٶ�
double* f0, * f1, * f2, * f3, * f4, * f5, * f6, * f7, * f8;	//�ֲ�����
//double* F0, * F1, * F2, * F3, * F4, * F5, * F6, * F7, * F8;	//������һʱ���ķֲ�����
int* Flag;												//���


int main()
{
	init();
	double error;

	double st = omp_get_wtime();

	for (int n = 0;; n++)
	{
		evolution(rho, ux, uy, u0x, u0y,
				  f0, f1, f2, f3, f4, f5, f6, f7, f8,
				  Flag);
		if (n % 100 == 0)
		{
			error = Error(ux, uy, u0x, u0y);
			cout << "The" << n << "th computation result:" << endl << "The u,v of point(NX/2,NY/2) is:" << setprecision(6) << ux[(NY / 2) * NX + (NX / 2)] << "," << uy[(NY / 2) * NX + (NX / 2)] << endl;
			cout << "The max relative error of uv is:" << setiosflags(ios::scientific) << error << endl;
			if (n >= 1000)
			{
				if (n % 1000 == 0) output(n);
				if (error < 3.0e-5) break;
			}
		}
	}

	double runtime = omp_get_wtime() - st;

	cout << "Total time is " << runtime << endl;

	memoryfinalize();

	return 0;
}

void init() //��ʼ��
{
	cout << "tau_f = " << tau_f << endl;

	memoryInitiate();

	for (int i = 0; i < NX; i++) //��ʼ��
		for (int j = 0; j < NY; j++)
		{
			ux[j * NX + i] = 0;
			uy[j * NX + i] = 0;
			//u0x[j * NX + i] = 0;
			//u0y[j * NX + i] = 0;

			rho[j * NX + i] = rho0;

			ux[(NY - 1) * NX + i] = U; //�����ٶ�

			//��ʼƽ��״̬
			f0[j * NX + i] = feq(0, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]);
			f1[j * NX + i] = feq(1, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]);
			f2[j * NX + i] = feq(2, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]);
			f3[j * NX + i] = feq(3, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]);
			f4[j * NX + i] = feq(4, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]);
			f5[j * NX + i] = feq(5, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]);
			f6[j * NX + i] = feq(6, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]);
			f7[j * NX + i] = feq(7, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]);
			f8[j * NX + i] = feq(8, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]);
		
			//��ʼ�����
			Flag[j * NX + i] = 'F';
			Flag[j * NX] = 'L';
			Flag[j * NX + NX - 1] = 'R';
			Flag[i] = 'B';
			Flag[(NY - 1) * NX + i] = 'T';
		}
	Flag[0] = 'M';
	Flag[NX - 1] = 'N';
	Flag[(NY - 1) * NX] = 'Q';
	Flag[NX * NY - 1] = 'P';
}

double feq(const int k, const double rho, const double ux, const double uy) //����ƽ��̬�ֲ�����
{
	double eu, uv;
	eu = e[k][0] * ux + e[k][1] * uy; //e_alpha*u
	uv = ux * ux + uy * uy; //u^2
	return w[k] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv);
}

void memoryInitiate() {

	size_t size = NX * NY * sizeof(double);
	size_t size_int = NX * NY * sizeof(int);

	rho = (double*)malloc(size);
	ux  = (double*)malloc(size);		uy  = (double*)malloc(size);
	u0x = (double*)malloc(size);		u0y = (double*)malloc(size);

	f0  = (double*)malloc(size);		//F0  = (double*)malloc(size);
	f1  = (double*)malloc(size);		//F1  = (double*)malloc(size);
	f2  = (double*)malloc(size);		//F2  = (double*)malloc(size);
	f3  = (double*)malloc(size);		//F3  = (double*)malloc(size);
	f4  = (double*)malloc(size);		//F4  = (double*)malloc(size);
	f5  = (double*)malloc(size);		//F5  = (double*)malloc(size);
	f6  = (double*)malloc(size);		//F6  = (double*)malloc(size);
	f7  = (double*)malloc(size);		//F7  = (double*)malloc(size);
	f8  = (double*)malloc(size);		//F8  = (double*)malloc(size);

	Flag = (int*)malloc(size_int);
}


void memoryfinalize() {

	free(rho);
	free(ux);		free(uy);
	free(u0x);		free(u0y);

	free(f0);		//free(F0);
	free(f1);		//free(F1);
	free(f2);		//free(F2);
	free(f3);		//free(F3);
	free(f4);		//free(F4);
	free(f5);		//free(F5);
	free(f6);		//free(F6);
	free(f7);		//free(F7);
	free(f8);		//free(F8);

	free(Flag);
}


void output(int m) //���
{
	ostringstream name;
	name << "cavity_" << m << ".dat";
	ofstream out(name.str().c_str());
	out << "Title=\"LBM Lid Driven Flow\"\n" << "VARIABLES = \"X\",\"Y\",\"U\",\"V\",\"P\"\n" << "ZONE T= \"BOX\",I= " << NX << ",J= " << NY << ",F=POINT" << endl;
	for (int j = 0; j < NY; j++)
		for (int i = 0; i < NX; i++)
		{
			out << i << " " << j << " " << ux[j * NX + i] << " " << uy[j * NX + i] << " " << rho[j * NX + i] / 3. << endl;
		}
}

double Error(const double* ux, const double* uy, const double* u0x, const double* u0y)
{
	double temp1, temp2;
	temp1 = 0;
	temp2 = 0;
	for (int i = 1; i < NX - 1; i++)
		for (int j = 1; j < NY - 1; j++)
		{
			temp1 += ((ux[j * NX + i] - u0x[j * NX + i]) * (ux[j * NX + i] - u0x[j * NX + i]) + (uy[j * NX + i] - u0y[j * NX + i]) * (uy[j * NX + i] - u0y[j * NX + i]));
			temp2 += (ux[j * NX + i] * ux[j * NX + i] + uy[j * NX + i] * uy[j * NX + i]);
		}
	temp1 = sqrt(temp1);
	temp2 = sqrt(temp2);
	return temp1 / (temp2 + 1e-30);
}
