#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <math.h>
#include "readWeights50.h"
#include "Functions50.h"
using namespace std;
int  main()
{
	double *X=(double *)malloc(1719551 * 3 * sizeof(double));//dataset
	double *W_i=(double *)malloc(50*3*sizeof(double));
	double *W_f=(double *)malloc(50*3*sizeof(double));
	double *W_c=(double *)malloc(50*3*sizeof(double));
	double *W_o=(double *)malloc(50*3*sizeof(double));
	double *U_i=(double *)malloc(50*50*sizeof(double));
	double *U_f=(double *)malloc(50*50*sizeof(double));
	double *U_c=(double *)malloc(50*50*sizeof(double));
	double *U_o=(double *)malloc(50*50*sizeof(double));
	double *b_i=(double *)malloc(50*sizeof(double));
	double *b_f=(double *)malloc(50*sizeof(double));
	double *b_c=(double *)malloc(50*sizeof(double));
	double *b_o=(double *)malloc(50*sizeof(double));
	double *w=(double *)malloc(100*12*sizeof(double));
	double *b=(double *)malloc(12*sizeof(double));
	double x[50][3];//input to lstm,50 timestamps
	double *c_t,*h_t,*i_t,*C_t,*f_t,*o_t;
	double H[50][100];//storing the output of each timestamp(50 timestamps, each output of size 50)
	double input[100],output[12];//input & output of dense layer
	double pd1[12],pd2[12];//probabbility density for upper and lower window resp.
	int z,i,j;
	double sum,res;
	ofstream fout;
	fout.open("results/bilstm_50_res.txt");
	readWeights(X,W_i,W_f,W_c,W_o,U_i,U_f,U_c,U_o,b_i,b_f,b_c,b_o,w,b);
	for(z=31908;z<=31968;z++)//selecting data point for prediction
	{
		//create upper window
		for(i=49;i>=0;i--)//i :timestamp from 0-49
		{
			x[i][0]=*(X+(z-(49-i))*3+0);
			x[i][1]=*(X+(z-(49-i))*3+1);
			x[i][2]=*(X+(z-(49-i))*3+2);
		}
		//prediction  for upper window
		for(i=0;i<50;i++)//i: timestamp(t)
		{
			if(i==0)
			{
				i_t=sigmoid(add(matmul1(W_i,x[i]),b_i));
				C_t=tan(add(matmul1(W_c,x[i]),b_c));
				f_t=sigmoid(add(matmul1(W_f,x[i]),b_f));
				c_t=mult(i_t,C_t);
				o_t=sigmoid(add(matmul1(W_o,x[i]),b_o));
				h_t=mult(o_t,tan(c_t));
				for(j=0;j<50;j++)
					H[i][j]=h_t[j];
				//if(z==31908)
				//cout<<H[i][0]<<'\n';
			}//if
			else
			{
				i_t=sigmoid(add(add(matmul1(W_i,x[i]),matmul2(U_i,h_t)),b_i));
				C_t=tan(add(add(matmul1(W_c,x[i]),matmul2(U_c,h_t)),b_c));
				f_t=sigmoid(add(add(matmul1(W_f,x[i]),matmul2(U_f,h_t)),b_f));
				c_t=add(mult(i_t,C_t),mult(f_t,c_t));
				o_t=sigmoid(add(add(matmul1(W_o,x[i]),matmul2(U_o,h_t)),b_o));
				h_t=mult(o_t,tan(c_t));
				for(j=0;j<50;j++)
					H[i][j]=h_t[j];
				//if(z==31908)
				//cout<<H[i][0]<<'\n';
			}//else
		}
		//backward pass
		for(i=49;i>=0;i--)//i :timestamp from 0-49
		{
			x[49-i][0]=*(X+(z-(49-i))*3+0);
			x[49-i][1]=*(X+(z-(49-i))*3+1);
			x[49-i][2]=*(X+(z-(49-i))*3+2);
		}
		for(i=0;i<50;i++)//i: timestamp(t)
		{
			if(i==0)
			{
				i_t=sigmoid(add(matmul1(W_i,x[i]),b_i));
				C_t=tan(add(matmul1(W_c,x[i]),b_c));
				f_t=sigmoid(add(matmul1(W_f,x[i]),b_f));
				c_t=mult(i_t,C_t);
				o_t=sigmoid(add(matmul1(W_o,x[i]),b_o));
				h_t=mult(o_t,tan(c_t));
				for(j=0;j<50;j++)
					H[i][50+j]=h_t[j];
			}//if
			else
			{
				i_t=sigmoid(add(add(matmul1(W_i,x[i]),matmul2(U_i,h_t)),b_i));
				C_t=tan(add(add(matmul1(W_c,x[i]),matmul2(U_c,h_t)),b_c));
				f_t=sigmoid(add(add(matmul1(W_f,x[i]),matmul2(U_f,h_t)),b_f));
				c_t=add(mult(i_t,C_t),mult(f_t,c_t));
				o_t=sigmoid(add(add(matmul1(W_o,x[i]),matmul2(U_o,h_t)),b_o));
				h_t=mult(o_t,tan(c_t));
				for(j=0;j<50;j++)
					H[i][50+j]=h_t[j];
			}//else
		}
		//Mean Pooling
		for(j=0;j<100;j++)
		{
			sum=0;
			for(i=0;i<50;i++)
				sum+=H[i][j];
			input[j]=sum/(50.0);
		}
		//Dense Layer
		sum=0;
		for(i=0;i<12;i++)
		{
			output[i]=b[i];
			for(j=0;j<100;j++)
				output[i]+=(input[j]*(*(w+j*12+i)));
			sum+=exp(output[i]);
		}
		for(i=0;i<12;i++)//prob density for upper window
			pd1[i]=exp(output[i])/sum;
		//create lower window
		for(i=0;i<50;i++)//i :timestamp from 0-49
		{
			x[i][0]=*(X+(z+i)*3+0);
			x[i][1]=*(X+(z+i)*3+1);
			x[i][2]=*(X+(z+i)*3+2);
		}
		//prediction  for lower window
		for(i=0;i<50;i++)//i: timestamp(t)
		{
			if(i==0)
			{
				i_t=sigmoid(add(matmul1(W_i,x[i]),b_i));
				C_t=tan(add(matmul1(W_c,x[i]),b_c));
				f_t=sigmoid(add(matmul1(W_f,x[i]),b_f));
				c_t=mult(i_t,C_t);
				o_t=sigmoid(add(matmul1(W_o,x[i]),b_o));
				h_t=mult(o_t,tan(c_t));
				for(j=0;j<50;j++)
					H[i][j]=h_t[j];
			}//if
			else
			{
				i_t=sigmoid(add(add(matmul1(W_i,x[i]),matmul2(U_i,h_t)),b_i));
				C_t=tan(add(add(matmul1(W_c,x[i]),matmul2(U_c,h_t)),b_c));
				f_t=sigmoid(add(add(matmul1(W_f,x[i]),matmul2(U_f,h_t)),b_f));
				c_t=add(mult(i_t,C_t),mult(f_t,c_t));
				o_t=sigmoid(add(add(matmul1(W_o,x[i]),matmul2(U_o,h_t)),b_o));
				h_t=mult(o_t,tan(c_t));
				for(j=0;j<50;j++)
					H[i][j]=h_t[j];
			}//else
		}
		//Backward pass
		for(i=0;i<50;i++)//i :timestamp from 0-49
		{
			x[49-i][0]=*(X+(z+i)*3+0);
			x[49-i][1]=*(X+(z+i)*3+1);
			x[49-i][2]=*(X+(z+i)*3+2);
		}
		//prediction  for lower window
		for(i=0;i<50;i++)//i: timestamp(t)
		{
			if(i==0)
			{
				i_t=sigmoid(add(matmul1(W_i,x[i]),b_i));
				C_t=tan(add(matmul1(W_c,x[i]),b_c));
				f_t=sigmoid(add(matmul1(W_f,x[i]),b_f));
				c_t=mult(i_t,C_t);
				o_t=sigmoid(add(matmul1(W_o,x[i]),b_o));
				h_t=mult(o_t,tan(c_t));
				for(j=0;j<50;j++)
					H[i][50+j]=h_t[j];
			}//if
			else
			{
				i_t=sigmoid(add(add(matmul1(W_i,x[i]),matmul2(U_i,h_t)),b_i));
				C_t=tan(add(add(matmul1(W_c,x[i]),matmul2(U_c,h_t)),b_c));
				f_t=sigmoid(add(add(matmul1(W_f,x[i]),matmul2(U_f,h_t)),b_f));
				c_t=add(mult(i_t,C_t),mult(f_t,c_t));
				o_t=sigmoid(add(add(matmul1(W_o,x[i]),matmul2(U_o,h_t)),b_o));
				h_t=mult(o_t,tan(c_t));
				for(j=0;j<50;j++)
					H[i][50+j]=h_t[j];
			}//else
		}
		//Mean Pooling
		for(j=0;j<100;j++)
		{
			sum=0;
			for(i=0;i<50;i++)
				sum+=H[i][j];
			input[j]=sum/(50.0);
		}
		//Dense Layer
		sum=0;
		for(i=0;i<12;i++)
		{
			output[i]=b[i];
			for(j=0;j<100;j++)
				output[i]+=(input[j]*(*(w+j*12+i)));
			sum+=exp(output[i]);
		}
		for(i=0;i<12;i++)//prob density for upper window
			pd2[i]=exp(output[i])/sum;
		res=0;
		for(i=0;i<12;i++)
			res+=(pd1[i]*pd2[i]);
		fout<<res<<'\n';
	}// data point loop
	fout.close();
	return 0;
}
