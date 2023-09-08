#ifndef LBM_H_ //�ļ���ͷ
	#define LBM_H_ 


void evolution()
{
	for (int i = 1; i < NX - 1; i++) //��ȥ���ұ߽�
		for (int j = 1; j < NY - 1; j++) //��ȥ���±߽�
			for (int k = 0; k < Q; k++)
			{
				int ip = i - e[k][0];
				int jp = j - e[k][1];
				F[i][j][k] = f[ip][jp][k] + (feq(k, rho[ip][jp], u[ip][jp]) - f[ip][jp][k]) / tau_f;
			}

	for (int i = 1; i < NX - 1; i++) //��������
		for (int j = 1; j < NY - 1; j++)
		{
			u0[i][j][0] = u[i][j][0];
			u0[i][j][1] = u[i][j][1];
			rho[i][j] = 0;
			u[i][j][0] = 0;
			u[i][j][1] = 0;
			for (int k = 0; k < Q; k++)
			{
				f[i][j][k] = F[i][j][k];
				rho[i][j] += f[i][j][k]; //����ܶ�
				u[i][j][0] += e[k][0] * f[i][j][k];
				u[i][j][1] += e[k][1] * f[i][j][k];
			}
			u[i][j][0] /= rho[i][j]; //����ٶ�
			u[i][j][1] /= rho[i][j];
		}
	//�߽紦����ƽ��̬���Ƹ�ʽ
	for (int j = 1; j < NY - 1; j++) //���ұ߽�
		for (int k = 0; k < Q; k++)
		{
			rho[NX - 1][j] = rho[NX - 2][j]; //�߽總��һ��ֵ���,�ұ߽�
			f[NX - 1][j][k] = feq(k, rho[NX - 1][j], u[NX - 1][j]) + f[NX - 2][j][k] - feq(k, rho[NX - 2][j], u[NX - 2][j]);
			rho[0][j] = rho[1][j]; //��߽�
			f[0][j][k] = feq(k, rho[0][j], u[0][j]) + f[1][j][k] - feq(k, rho[1][j], u[1][j]);
		}
	for (int i = 0; i < NX; i++) //���±߽�
		for (int k = 0; k < Q; k++)
		{
			rho[i][0] = rho[i][1]; //�±߽�
			f[i][0][k] = feq(k, rho[i][0], u[i][0]) + f[i][1][k] - feq(k, rho[i][1], u[i][1]);
			rho[i][NY - 1] = rho[i][NY - 2]; //�ϱ߽�
			u[i][NY - 1][0] = U;
			f[i][NY - 1][k] = feq(k, rho[i][NY - 1], u[i][NY - 1]) + f[i][NY - 2][k] - feq(k, rho[i][NY - 2], u[i][NY - 2]);
		}
}

#endif