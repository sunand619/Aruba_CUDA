#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>
#include <ctime>
#include <sys/time.h>
#include "readWeights30.h"//to read the weights
#include "deviceFunctions30.h"//contains device functions like matmul,add
using namespace std;
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        {   -1, -1 }
    };
	    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}
__global__ void predictKernel(double *X,double *W_i,double *W_f,double *W_c,double *W_o,double *U_i,double *U_f,double *U_c,double *U_o,double *b_i,double *b_f,double *b_c,double *b_o,double *w,double *b,double *result,double *loop_count)//cuda kernel
	{
		// Get our global thread ID
		int tid = blockIdx.x*blockDim.x+threadIdx.x;
		//if(tid==31908)
		//printf("Done");
		loop_count[0]=0;
		double x[30][3];//input to lstm,50 timestamps
		double *c_t,*h_t,*i_t,*C_t,*f_t,*o_t;
		double H[30][60];//storing the output of each timestamp(50 timestamps, each output of size 50)
		double input[60],output[12];//input & output of dense layer
		double pd1[12],pd2[12];//probabbility density for upper and lower window resp.
		int i,j;
		double sum,res;
		if ((tid>29&&tid<1719551-30))
		{
			//create upper window
			#pragma unroll
			for(i=29;i>=0;i--)//i :timestamp from 0-49
			{
				x[i][0]=*(X+(tid-(29-i))*3+0);
				x[i][1]=*(X+(tid-(29-i))*3+1);
				x[i][2]=*(X+(tid-(29-i))*3+2);
				loop_count[0]++;
			}
			//prediction  for upper window
			#pragma unroll
			for(i=0;i<30;i++)//i: timestamp(t)
			{
				if(i==0)
				{
					i_t=sigmoid(add(matmul1(W_i,x[i]),b_i));
					C_t=tan(add(matmul1(W_c,x[i]),b_c));
					f_t=sigmoid(add(matmul1(W_f,x[i]),b_f));
					c_t=mult(i_t,C_t);
					o_t=sigmoid(add(matmul1(W_o,x[i]),b_o));
					h_t=mult(o_t,tan(c_t));
					
                        		#pragma unroll
					for(j=0;j<30;j++)
					{
					H[i][j]=h_t[j];
					loop_count[0]++;
					}
				}//if
				else
				{
					i_t=sigmoid(add(add(matmul1(W_i,x[i]),matmul2(U_i,h_t)),b_i));
					C_t=tan(add(add(matmul1(W_c,x[i]),matmul2(U_c,h_t)),b_c));
					f_t=sigmoid(add(add(matmul1(W_f,x[i]),matmul2(U_f,h_t)),b_f));
					c_t=add(mult(i_t,C_t),mult(f_t,c_t));
					o_t=sigmoid(add(add(matmul1(W_o,x[i]),matmul2(U_o,h_t)),b_o));
					h_t=mult(o_t,tan(c_t));
					#pragma unroll
					for(j=0;j<30;j++)
					{
						H[i][j]=h_t[j];
						loop_count[0]++;
					}
				}//else
			}
			//backward pass
			#pragma unroll
			 for(i=29;i>=0;i--)//i :timestamp from 0-49
                	{
                        x[29-i][0]=*(X+(tid-(29-i))*3+0);
                        x[29-i][1]=*(X+(tid-(29-i))*3+1);
                        x[29-i][2]=*(X+(tid-(29-i))*3+2);
			loop_count[0]++;
                	}
			#pragma unroll
			for(i=0;i<30;i++)//i: timestamp(t)
                        {
                                if(i==0)
                                {
                                        i_t=sigmoid(add(matmul1(W_i,x[i]),b_i));
                                        C_t=tan(add(matmul1(W_c,x[i]),b_c));
                                        f_t=sigmoid(add(matmul1(W_f,x[i]),b_f));
                                        c_t=mult(i_t,C_t);
                                        o_t=sigmoid(add(matmul1(W_o,x[i]),b_o));
                                        h_t=mult(o_t,tan(c_t));
					
                                        #pragma unroll
                                        for(j=0;j<30;j++)
					{
                                        H[i][30+j]=h_t[j];
					loop_count[0]++;
					}
                                }//if
                                else
                                {
                                        i_t=sigmoid(add(add(matmul1(W_i,x[i]),matmul2(U_i,h_t)),b_i));
                                        C_t=tan(add(add(matmul1(W_c,x[i]),matmul2(U_c,h_t)),b_c));
                                        f_t=sigmoid(add(add(matmul1(W_f,x[i]),matmul2(U_f,h_t)),b_f));
                                        c_t=add(mult(i_t,C_t),mult(f_t,c_t));
                                        o_t=sigmoid(add(add(matmul1(W_o,x[i]),matmul2(U_o,h_t)),b_o));
                                        h_t=mult(o_t,tan(c_t));
                                        #pragma unroll
                                        for(j=0;j<30;j++)
					{
                                                H[i][30+j]=h_t[j];
						loop_count[0]++;
					}
                                }//else
                        }

			//Mean Pooling
			#pragma unroll
			for(j=0;j<60;j++)
			{
				sum=0;
				#pragma unroll
				for(i=0;i<30;i++)
				{
					sum+=H[i][j];
					loop_count[0]++;
				}
				input[j]=sum/(30.0);
			}
			//Dense Layer
			sum=0;
			#pragma unroll
			for(i=0;i<12;i++)
			{
				output[i]=b[i];
				#pragma unroll
				for(j=0;j<60;j++)
				{
					output[i]+=(input[j]*(*(w+j*12+i)));
					loop_count[0]++;
				}
				sum+=exp(output[i]);
			}
			#pragma unroll
			for(i=0;i<12;i++)//prob density for upper window
			{
				pd1[i]=exp(output[i])/sum;
				loop_count[0]++;
			}
			//create lower window
			#pragma unroll
			for(i=0;i<30;i++)//i :timestamp from 0-49
			{
				x[i][0]=*(X+(tid+i)*3+0);
				x[i][1]=*(X+(tid+i)*3+1);
				x[i][2]=*(X+(tid+i)*3+2);
				loop_count[0]++;
			}
			//prediction  for lower window
			#pragma unroll
			for(i=0;i<30;i++)//i: timestamp(t)
			{

				if(i==0)
				{
					i_t=sigmoid(add(matmul1(W_i,x[i]),b_i));
					C_t=tan(add(matmul1(W_c,x[i]),b_c));
					f_t=sigmoid(add(matmul1(W_f,x[i]),b_f));
					c_t=mult(i_t,C_t);
					o_t=sigmoid(add(matmul1(W_o,x[i]),b_o));
					h_t=mult(o_t,tan(c_t));
					#pragma unroll
					for(j=0;j<30;j++)
					{
						H[i][j]=h_t[j];
						loop_count[0]++;
					}
				}//if
				else
				{
					i_t=sigmoid(add(add(matmul1(W_i,x[i]),matmul2(U_i,h_t)),b_i));
					C_t=tan(add(add(matmul1(W_c,x[i]),matmul2(U_c,h_t)),b_c));
					f_t=sigmoid(add(add(matmul1(W_f,x[i]),matmul2(U_f,h_t)),b_f));
					c_t=add(mult(i_t,C_t),mult(f_t,c_t));
					o_t=sigmoid(add(add(matmul1(W_o,x[i]),matmul2(U_o,h_t)),b_o));
					h_t=mult(o_t,tan(c_t));
					#pragma unroll
					for(j=0;j<30;j++)
					{
						H[i][j]=h_t[j];
						loop_count[0]++;
					}
				}//else
			}
		//Backward pass
		#pragma unroll
                for(i=0;i<30;i++)//i :timestamp from 0-49
                {
                        x[29-i][0]=*(X+(tid+i)*3+0);
                        x[29-i][1]=*(X+(tid+i)*3+1);
                        x[29-i][2]=*(X+(tid+i)*3+2);
			loop_count[0]++;
                }
                //prediction  for lower window
		#pragma unroll
                for(i=0;i<30;i++)//i: timestamp(t)
                {
                        if(i==0)
                        {
                                i_t=sigmoid(add(matmul1(W_i,x[i]),b_i));
                                C_t=tan(add(matmul1(W_c,x[i]),b_c));
                                f_t=sigmoid(add(matmul1(W_f,x[i]),b_f));
                                c_t=mult(i_t,C_t);
                                o_t=sigmoid(add(matmul1(W_o,x[i]),b_o));
                                h_t=mult(o_t,tan(c_t));
				#pragma unroll
                                for(j=0;j<30;j++)
				{
                                        H[i][30+j]=h_t[j];
					loop_count[0]++;
				}
                        }//if
                        else
                        {
                                i_t=sigmoid(add(add(matmul1(W_i,x[i]),matmul2(U_i,h_t)),b_i));
                                C_t=tan(add(add(matmul1(W_c,x[i]),matmul2(U_c,h_t)),b_c));
                                f_t=sigmoid(add(add(matmul1(W_f,x[i]),matmul2(U_f,h_t)),b_f));
                                c_t=add(mult(i_t,C_t),mult(f_t,c_t));
                                o_t=sigmoid(add(add(matmul1(W_o,x[i]),matmul2(U_o,h_t)),b_o));
                                h_t=mult(o_t,tan(c_t));
				#pragma unroll
                                for(j=0;j<30;j++)
				{
                                        H[i][30+j]=h_t[j];
					loop_count[0]++;
				}
                        }//else
                }
			//Mean Pooling
			#pragma unroll
			for(j=0;j<60;j++)
			{
				sum=0;
				#pragma unroll
				for(i=0;i<30;i++)
				{
					sum+=H[i][j];
					loop_count[0]++;
				}
				input[j]=sum/(30.0);
			}
			//Dense Layer
			sum=0;
			#pragma unroll
			for(i=0;i<12;i++)
			{
				output[i]=b[i];
				#pragma unroll
				for(j=0;j<60;j++)
				{
					output[i]+=(input[j]*(*(w+j*12+i)));
					loop_count[0]++;
				}
				sum+=exp(output[i]);
			}
			#pragma unroll
			for(i=0;i<12;i++)//prob density for upper window
			{
				pd2[i]=exp(output[i])/sum;
				loop_count[0]++;
			}

			res=0;
			#pragma unroll
			for(i=0;i<12;i++)
			{
				res+=(pd1[i]*pd2[i]);
				loop_count[0]++;
			}
			*(result+tid)=res;
		}//if tid
	}// kernel loop
int main()
{
        double *X=(double *)malloc(1719551 * 3 * sizeof(double));//dataset
	double *W_i=(double *)malloc(30*3*sizeof(double));
	double *W_f=(double *)malloc(30*3*sizeof(double));
	double *W_c=(double *)malloc(30*3*sizeof(double));
	double *W_o=(double *)malloc(30*3*sizeof(double));
	double *U_i=(double *)malloc(30*30*sizeof(double));
	double *U_f=(double *)malloc(30*30*sizeof(double));
	double *U_c=(double *)malloc(30*30*sizeof(double));
	double *U_o=(double *)malloc(30*30*sizeof(double));
	double *b_i=(double *)malloc(30*sizeof(double));
	double *b_f=(double *)malloc(30*sizeof(double));
	double *b_c=(double *)malloc(30*sizeof(double));
	double *b_o=(double *)malloc(30*sizeof(double));
	double *w=(double *)malloc(60*12*sizeof(double));
	double *b=(double *)malloc(12*sizeof(double));
	double *result=(double *)malloc(1719551*sizeof(double));
	double *loop_count=(double *)malloc(1*sizeof(double));
	readWeights(X,W_i,W_f,W_c,W_o,U_i,U_f,U_c,U_o,b_i,b_f,b_c,b_o,w,b);//read the weights from file(readWeights.h)
	//for(int p=0;p<50;p++)
	//printf("%f ",*(b_i+p));
	//printf("\n");
	double *X_gpu,*W_i_gpu,*W_f_gpu,*W_c_gpu,*W_o_gpu,*U_i_gpu,*U_f_gpu,*U_c_gpu,*U_o_gpu,*b_i_gpu,*b_f_gpu,*b_c_gpu,*b_o_gpu,*w_gpu,*b_gpu,*result_gpu,*loop_count_gpu;//device vector
	
	size_t bytes1=1719551*3*sizeof(double);//size in bytes of the vector to be sent to gpu
	size_t bytes2=30*3*sizeof(double);
	size_t bytes3=30*30*sizeof(double);
	size_t bytes4=30*sizeof(double);
	size_t bytes5=60*12*sizeof(double);
	size_t bytes6=12*sizeof(double);
	size_t bytes7=1719551*sizeof(double);
	// Allocate memory for each vector on GPU
	cudaMalloc(&X_gpu, bytes1);
	cudaMalloc(&W_i_gpu,bytes2);
	cudaMalloc(&W_f_gpu,bytes2);
	cudaMalloc(&W_c_gpu,bytes2);
	cudaMalloc(&W_o_gpu,bytes2);
	cudaMalloc(&U_i_gpu,bytes3);
	cudaMalloc(&U_f_gpu,bytes3);
	cudaMalloc(&U_c_gpu,bytes3);
	cudaMalloc(&U_o_gpu,bytes3);
	cudaMalloc(&b_i_gpu,bytes4);
	cudaMalloc(&b_f_gpu,bytes4);
	cudaMalloc(&b_c_gpu,bytes4);
	cudaMalloc(&b_o_gpu,bytes4);
	cudaMalloc(&w_gpu,bytes5);
	cudaMalloc(&b_gpu,bytes6);
	cudaMalloc(&result_gpu,bytes7);
	cudaMalloc(&loop_count_gpu,1*sizeof(double));
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Copy host vectors to device
	cudaMemcpy(X_gpu,X,bytes1,cudaMemcpyHostToDevice);
	cudaMemcpy(W_i_gpu,W_i,bytes2,cudaMemcpyHostToDevice);
	cudaMemcpy(W_f_gpu,W_f,bytes2,cudaMemcpyHostToDevice);
	cudaMemcpy(W_c_gpu,W_c,bytes2,cudaMemcpyHostToDevice);
	cudaMemcpy(W_o_gpu,W_o,bytes2,cudaMemcpyHostToDevice);
	cudaMemcpy(U_i_gpu,U_i,bytes3,cudaMemcpyHostToDevice);
	cudaMemcpy(U_f_gpu,U_f,bytes3,cudaMemcpyHostToDevice);
	cudaMemcpy(U_c_gpu,U_c,bytes3,cudaMemcpyHostToDevice);
	cudaMemcpy(U_o_gpu,U_o,bytes3,cudaMemcpyHostToDevice);
	cudaMemcpy(b_i_gpu,b_i,bytes4,cudaMemcpyHostToDevice);
	cudaMemcpy(b_f_gpu,b_f,bytes4,cudaMemcpyHostToDevice);
	cudaMemcpy(b_c_gpu,b_c,bytes4,cudaMemcpyHostToDevice);
	cudaMemcpy(b_o_gpu,b_o,bytes4,cudaMemcpyHostToDevice);
	cudaMemcpy(w_gpu,w,bytes5,cudaMemcpyHostToDevice);
	cudaMemcpy(b_gpu,b,bytes6,cudaMemcpyHostToDevice);
	cudaMemcpy(loop_count_gpu,loop_count,1*sizeof(double),cudaMemcpyHostToDevice);
	int blockSize, gridSize;
	// Number of threads in each thread block
	blockSize = 1024;
	// Number of thread blocks in grid
	gridSize = (int)ceil((float)1719551/blockSize);
		// Execute the kernel
  	//Gflops
	double fs_t, fe_t, ft_t;
	struct timeval t;
	int  cudaCores, smCount, totalThreads;
	double f_avg;
	int i=0;
	cudaSetDevice(i);
	// Get device properties
	printf("\nCUDA Device #%d\n\n", (i+1));
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, i);
	smCount = devProp.multiProcessorCount;
	cudaCores = _ConvertSMVer2Cores(devProp.major, devProp.minor);
	totalThreads=1719551-60;
	gettimeofday(&t, NULL);
	fs_t = t.tv_sec+(t.tv_usec/1000000.0);
	cudaEventRecord(start);
	predictKernel<<<gridSize, blockSize>>>(X_gpu,W_i_gpu,W_f_gpu,W_c_gpu,W_o_gpu,U_i_gpu,U_f_gpu,U_c_gpu,U_o_gpu,b_i_gpu,b_f_gpu,b_c_gpu,b_o_gpu,w_gpu,b_gpu,result_gpu,loop_count_gpu);
	cudaEventRecord(stop);
		
	cudaThreadSynchronize();
	gettimeofday(&t, NULL);
	fe_t = t.tv_sec+(t.tv_usec/1000000.0);
	ft_t = fe_t - fs_t;
	cudaMemcpy(loop_count,loop_count_gpu,sizeof(double),cudaMemcpyDeviceToHost);
	cout<<loop_count[0]<<' '<<smCount<<' '<<cudaCores<<' '<<totalThreads<<'\n';
	
	f_avg += (loop_count[0]*smCount*cudaCores*totalThreads*10)/(ft_t*1000000000);

	cudaMemcpy(result,result_gpu,bytes7,cudaMemcpyDeviceToHost);
        cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
  	cout<<"Time:"<<'\n';
	cout<<(float)(milliseconds/1000)<<'\n';
	for(int z=31908;z<=31968;z++)
	cout<<result[z]<<' ';
	printf("Number of FLOPs: %lf G-FLOPs\n", (f_avg));
        cudaFree(X_gpu);
        cudaFree(W_i_gpu);
        cudaFree(W_f_gpu);
        cudaFree(W_c_gpu);
        cudaFree(W_o_gpu);
        cudaFree(U_i_gpu);
        cudaFree(U_f_gpu);
        cudaFree(U_c_gpu);
        cudaFree(U_o_gpu);
        cudaFree(b_i_gpu);
        cudaFree(b_f_gpu);
        cudaFree(b_c_gpu);
        cudaFree(b_o_gpu);
        cudaFree(w_gpu);
        cudaFree(b_gpu);
	cudaFree(result_gpu);
	return 0;
}
