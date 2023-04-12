#include<stdio.h>
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		C[i] = A[i] + B[i];
}
int main()
{
	int N =1024;
	size_t size = N * sizeof(float);
	float* h_A = (float *)malloc(size);
	float* h_B = (float *)malloc(size);
	float* h_C = (float *)malloc(size);
	int i=0;
	// Initialize input vectors
	for(i=0;i<N;i++)
	{
		h_A[i]=i;
		h_B[i]=2*i;
		h_C[i]=0;
	} 

	for(i=0;i<N;i++)
	{
		printf("%d ",h_A[i]); 
	} 
	// Allocate vectors in device memory
	float* d_A;
	cudaMalloc(&d_A, size);
	float* d_B;
	cudaMalloc(&d_B, size);
	float* d_C;
	cudaMalloc(&d_C, size);
	// Copy vectors from host memory to device memory
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	// Invoke kernel
	int threadsPerBlock = 256;
	int blocksPerGrid =
	(N + threadsPerBlock - 1) / threadsPerBlock;
	VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	// Copy result from device memory to host memory
	// h_C contains the result in host memory
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	for(i=0;i<N;i++)
	{
		printf("%d ",h_C[i]);
	} 
	
	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
