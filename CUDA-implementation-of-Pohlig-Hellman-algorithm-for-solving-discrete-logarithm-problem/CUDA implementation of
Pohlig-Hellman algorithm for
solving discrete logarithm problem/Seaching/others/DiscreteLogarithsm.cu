#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h> 

int * myloadFile(int n,char *s){
	FILE *f=fopen(s,"r");
	int i;
	int *a=(int *)malloc(sizeof(int)*n);
	for(i=0;i<n;i++)
	{
		int x;
		fscanf(f,"%d",&x);
		a[i]=x;
	}
	return a;
	
} 
void display(int *a,int n){
	int i;
	for(i=0;i<n;i++)
	{
		printf("%d %d \n",i,a[i]);
 	}
		 


}
 

__global__ void addKernel(int* A, int* B, int* C, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
	C[i] = A[i] + B[i];
}
__global__ void displayDemo(int* A, int* B, int* C, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
 
	printf("%d (blockId:%d,Thread Id:%d)\n",i, blockIdx.x,threadIdx.x);
}
 
 
int main()
{
	int N =4096;
	size_t size = N * sizeof(int);
 
	int* h_A = myloadFile(N,"input1.txt");
	int* h_B = myloadFile(N,"input2.txt");
 	int* h_C = (int*)malloc(size);
	int* d_A;
	cudaMalloc(&d_A, size);
	int* d_B;
	cudaMalloc(&d_B, size);
	int* d_C;
	cudaMalloc(&d_C, size);
	// Copy vectors from host memory to device memory
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	// Invoke kernel
	int threadsPerBlock = 256;
	int blocksPerGrid =
	(N + threadsPerBlock - 1) / threadsPerBlock;
	displayDemo<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	// Copy result from device memory to host memory
	// h_C contains the result in host memory
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	//display(h_C,N);
	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
 
