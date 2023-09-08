
#include "basicVar.h"

void evolution(double* rho, double* ux, double* uy, double* u0x, double* u0y,
			   double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
			   double* F0, double* F1, double* F2, double* F3, double* F4, double* F5, double* F6, double* F7, double* F8)
{

	/*
	����˳������ǣ�
	1. ��ײ
	2. �߽�����
	3. Ǩ��
	4. �����
	*/

	/*
	����˳����Ϊ ��ײ �� Ǩ�� �� ����� �� �߽����� ��ط�ɢ
	��Ϊ��ԭʼ�汾�У�F[i][j][k] = f[ip][jp][k] + (feq(k, rho[ip][jp], u[ip][jp]) - f[ip][jp][k]) / tau_f; �Ⱥ��ұ߽��߽�һ��������
	���ڱ��������ײ�����У���û�м���߽��ϵķֲ�����
	*/
	for (int i = 1; i < NX - 1; i++) //��ȥ���ұ߽�
		for (int j = 1; j < NY - 1; j++) //��ȥ���±߽�
		{
			f0[j * NX + i] = f0[j * NX + i] + (feq(0, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]) - f0[j * NX + i]) / tau_f;
			f1[j * NX + i] = f1[j * NX + i] + (feq(1, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]) - f1[j * NX + i]) / tau_f;
			f2[j * NX + i] = f2[j * NX + i] + (feq(2, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]) - f2[j * NX + i]) / tau_f;
			f3[j * NX + i] = f3[j * NX + i] + (feq(3, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]) - f3[j * NX + i]) / tau_f;
			f4[j * NX + i] = f4[j * NX + i] + (feq(4, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]) - f4[j * NX + i]) / tau_f;
			f5[j * NX + i] = f5[j * NX + i] + (feq(5, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]) - f5[j * NX + i]) / tau_f;
			f6[j * NX + i] = f6[j * NX + i] + (feq(6, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]) - f6[j * NX + i]) / tau_f;
			f7[j * NX + i] = f7[j * NX + i] + (feq(7, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]) - f7[j * NX + i]) / tau_f;
			f8[j * NX + i] = f8[j * NX + i] + (feq(8, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]) - f8[j * NX + i]) / tau_f;
			
			//F0[j * NX + i] = f0[j * NX + i		] + (feq(0, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]) - f0[j * NX + i]) / tau_f;
			//F1[j * NX + i] = f1[j * NX + i - 1	] + (feq(1, rho[j * NX + i - 1], ux[j * NX + i - 1], uy[j * NX + i - 1]) - f1[j * NX + i - 1]) / tau_f;
			//F2[j * NX + i] = f2[(j - 1) * NX + i] + (feq(2, rho[(j - 1) * NX + i], ux[(j - 1) * NX + i], uy[(j - 1) * NX + i]) - f2[(j - 1) * NX + i]) / tau_f;
			//F3[j * NX + i] = f3[j * NX + i + 1  ] + (feq(3, rho[j * NX + i + 1], ux[j * NX + i + 1], uy[j * NX + i + 1]) - f3[j * NX + i + 1]) / tau_f;
			//F4[j * NX + i] = f4[(j + 1) * NX + i] + (feq(4, rho[(j + 1) * NX + i], ux[(j + 1) * NX + i], uy[(j + 1) * NX + i]) - f4[(j + 1) * NX + i]) / tau_f;
			//F5[j * NX + i] = f5[(j - 1) * NX + i - 1] + (feq(5, rho[(j - 1) * NX + i - 1], ux[(j - 1) * NX + i - 1], uy[(j - 1) * NX + i - 1]) - f5[(j - 1) * NX + i - 1]) / tau_f;
			//F6[j * NX + i] = f6[(j - 1) * NX + i + 1] + (feq(6, rho[(j - 1) * NX + i + 1], ux[(j - 1) * NX + i + 1], uy[(j - 1) * NX + i + 1]) - f6[(j - 1) * NX + i + 1]) / tau_f;
			//F7[j * NX + i] = f7[(j + 1) * NX + i + 1] + (feq(7, rho[(j + 1) * NX + i + 1], ux[(j + 1) * NX + i + 1], uy[(j + 1) * NX + i + 1]) - f7[(j + 1) * NX + i + 1]) / tau_f;
			//F8[j * NX + i] = f8[(j + 1) * NX + i - 1] + (feq(8, rho[(j + 1) * NX + i - 1], ux[(j + 1) * NX + i - 1], uy[(j + 1) * NX + i - 1]) - f8[(j + 1) * NX + i - 1]) / tau_f;
		}

	//�߽紦����ƽ��̬���Ƹ�ʽ
	for (int j = 1; j < NY - 1; j++) //���ұ߽�
	{
		//��߽�
		//�������ܶȵĲ�ֵ
		rho[j * NX] = rho[j * NX + 1];
		//ux[j * NX] = 0;�ٶȿ��п���
		//uy[j * NX] = 0;
		f1[j * NX] = feq(1, rho[1 + j * NX], 0, 0) + f1[1 + j * NX] - feq(1, rho[1 + j * NX], ux[1 + j * NX], uy[1 + j * NX]);
		f5[j * NX] = feq(5, rho[1 + j * NX], 0, 0) + f5[1 + j * NX] - feq(5, rho[1 + j * NX], ux[1 + j * NX], uy[1 + j * NX]);
		f8[j * NX] = feq(8, rho[1 + j * NX], 0, 0) + f8[1 + j * NX] - feq(8, rho[1 + j * NX], ux[1 + j * NX], uy[1 + j * NX]);

		//�ұ߽�
		rho[j * NX + NX - 1] = rho[j * NX + NX - 2];
		//ux[j * NX + NX - 1] = 0;
		//uy[j * NX + NX - 1] = 0;
		f3[j * NX + NX - 1] = feq(3, rho[j * NX + NX - 2], 0, 0) + f3[j * NX + NX - 2] - feq(3, rho[j * NX + NX - 2], ux[j * NX + NX - 2], uy[j * NX + NX - 2]);
		f6[j * NX + NX - 1] = feq(6, rho[j * NX + NX - 2], 0, 0) + f6[j * NX + NX - 2] - feq(6, rho[j * NX + NX - 2], ux[j * NX + NX - 2], uy[j * NX + NX - 2]);
		f7[j * NX + NX - 1] = feq(7, rho[j * NX + NX - 2], 0, 0) + f7[j * NX + NX - 2] - feq(7, rho[j * NX + NX - 2], ux[j * NX + NX - 2], uy[j * NX + NX - 2]);
	}
	for (int i = 0; i < NX; i++) //���±߽�
	{
		//�ϱ߽�
		rho[(NY - 1) * NX + i] = rho[(NY - 2) * NX + i];
		ux[(NY - 1) * NX + i] = U;
		//uy[(NY - 1) * NX + i] = 0;
		f4[i + NX * (NY - 1)] = feq(4, rho[i + NX * (NY - 2)], U, 0) + f4[i + NX * (NY - 2)] - feq(4, rho[i + NX * (NY - 2)], ux[i + NX * (NY - 2)], uy[i + NX * (NY - 2)]);
		f7[i + NX * (NY - 1)] = feq(7, rho[i + NX * (NY - 2)], U, 0) + f7[i + NX * (NY - 2)] - feq(7, rho[i + NX * (NY - 2)], ux[i + NX * (NY - 2)], uy[i + NX * (NY - 2)]);
		f8[i + NX * (NY - 1)] = feq(8, rho[i + NX * (NY - 2)], U, 0) + f8[i + NX * (NY - 2)] - feq(8, rho[i + NX * (NY - 2)], ux[i + NX * (NY - 2)], uy[i + NX * (NY - 2)]);

		//�±߽�
		rho[i] = rho[NX + i];
		//ux[i] = 0;
		//uy[i] = 0;
		f2[i] = feq(2, rho[i + NX], 0, 0) + f2[i + NX] - feq(2, rho[i + NX], ux[i + NX], uy[i + NX]);
		f5[i] = feq(5, rho[i + NX], 0, 0) + f5[i + NX] - feq(5, rho[i + NX], ux[i + NX], uy[i + NX]);
		f6[i] = feq(6, rho[i + NX], 0, 0) + f6[i + NX] - feq(6, rho[i + NX], ux[i + NX], uy[i + NX]);
	}

	//�ĸ�����
	//f5[0] = feq(5, rho[NX + 1], 0, 0) + f5[NX + 1] - feq(5, rho[NX + 1], ux[NX + 1], uy[NX + 1]); //���½�
	//f6[NX - 1] = feq(6, rho[2 * NX - 2], 0, 0) + f6[2 * NX - 2] - feq(6, rho[2 * NX - 2], ux[2 * NX - 2], uy[2 * NX - 2]); //���½�
	//f8[(NY - 1) * NX] = feq(8, rho[(NY - 2) * NX + 1], U, 0) + f8[(NY - 2) * NX + 1] - feq(8, rho[(NY - 2) * NX + 1], ux[(NY - 2) * NX + 1], uy[(NY - 2) * NX + 1]);
	//f7[NY * NX - 1] = feq(7, rho[(NY - 1) * NX - 2], U, 0) + f7[(NY - 1) * NX - 2] - feq(7, rho[(NY - 1) * NX - 2], ux[(NY - 1) * NX - 2], uy[(NY - 1) * NX - 2]);

	for (int i = 1; i < NX - 1; ++i)
		for (int j = 1; j < NY - 1; ++j) {
	
			F0[j * NX + i] = f0[j * NX + i];
			F1[j * NX + i] = f1[j * NX + i - 1];
			F2[j * NX + i] = f2[(j - 1) * NX + i];
			F3[j * NX + i] = f3[j * NX + i + 1];
			F4[j * NX + i] = f4[(j + 1) * NX + i];
			F5[j * NX + i] = f5[(j - 1) * NX + i - 1];
			F6[j * NX + i] = f6[(j - 1) * NX + i + 1];
			F7[j * NX + i] = f7[(j + 1) * NX + i + 1];
			F8[j * NX + i] = f8[(j + 1) * NX + i - 1];
		}

	for (int i = 1; i < NX - 1; i++) //��������
		for (int j = 1; j < NY - 1; j++)
		{
			u0x[j * NX + i] = ux[j * NX + i];
			u0y[j * NX + i] = uy[j * NX + i];

			f0[j * NX + i] = F0[j * NX + i];
			f1[j * NX + i] = F1[j * NX + i];
			f2[j * NX + i] = F2[j * NX + i];
			f3[j * NX + i] = F3[j * NX + i];
			f4[j * NX + i] = F4[j * NX + i];
			f5[j * NX + i] = F5[j * NX + i];
			f6[j * NX + i] = F6[j * NX + i];
			f7[j * NX + i] = F7[j * NX + i];
			f8[j * NX + i] = F8[j * NX + i];

			//����ܶ�
			rho[j * NX + i] = f0[j * NX + i] + f1[j * NX + i] + f2[j * NX + i] + f3[j * NX + i] + f4[j * NX + i] + f5[j * NX + i] + f6[j * NX + i] + f7[j * NX + i] + f8[j * NX + i];
			ux[j * NX + i] = (f1[j * NX + i] - f3[j * NX + i] + f5[j * NX + i] - f6[j * NX + i] - f7[j * NX + i] + f8[j * NX + i]) / rho[j * NX + i];
			uy[j * NX + i] = (f2[j * NX + i] - f4[j * NX + i] + f5[j * NX + i] + f6[j * NX + i] - f7[j * NX + i] - f8[j * NX + i]) / rho[j * NX + i];

		}
	
}