#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <math.h>
#include <stdlib.h>
using namespace std;
__device__ double *matmul1(double *m1,double *m2)
{
	int r1=30,c1=3;
	// double *m = (double *)malloc(50 * sizeof(double));
	double m[30];
	for(int i=0;i<r1;i++)
	{
		double sum=0;
		for(int j=0;j<c1;j++)
			sum+=((*(m1+i*3+j))*(*(m2+j)));
		m[i]=sum;
	}
	return m;
}
__device__ double *matmul2(double *m1,double *m2)
{
	int r1=30,c1=30;
	// double *m = (double *)malloc(50 * sizeof(double));
	double m[30];
	for(int i=0;i<r1;i++)
	{
		double sum=0;
		for(int j=0;j<c1;j++)
			sum+=((*(m1+i*30+j))*(*(m2+j)));

		m[i]=sum;
	}
	return m;
}
__device__ double *add(double *m1,double *m2)
{
	//double *m = (double *)malloc(50 * sizeof(double));
	double m[30];		
	for(int i=0;i<30;i++)
		m[i]=m1[i]+m2[i];

	return m;
}
__device__ double *sigmoid(double *m1)
{
	//double *m = (double *)malloc(50 * sizeof(double));
	double m[30];
	for(int i=0;i<30;i++)
		m[i]=1/(1+exp(-m1[i]));
	return m;
}
__device__ double *tan(double *m1)
{
	//double *m = (double *)malloc(50 * sizeof(double));
	double m[30];			
	for(int i=0;i<30;i++)
		m[i]=tanh(m1[i]);
	return m;
}
__device__ double *mult(double *m1,double *m2)
{
	//double *m = (double *)malloc(50 * sizeof(double));
	double m[30];	
	for(int i=0;i<30;i++)
		m[i]=m1[i]*m2[i];
	return m;
}

