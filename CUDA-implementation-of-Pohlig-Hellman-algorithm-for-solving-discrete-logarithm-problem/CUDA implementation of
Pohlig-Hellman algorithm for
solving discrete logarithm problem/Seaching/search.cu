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

__global__ void searchKernel(int *a, int *element,int *d_index)
{
	int search_ele=element[0];
	int index;
	int tid =blockIdx.x * blockDim.x + threadIdx.x;//threadIdx.x;
	int low=677*tid;
	int high=677*(tid+1)-1;	
	for(int i=low;i<=high;i++){
		if(search_ele==a[i]){
			index=i;
			d_index[0]=i;
			printf("\nindex:%d",index);
			break;

		}
	}
}
// Helper function for using CUDA to add vectors in parallel.
cudaError_t searchWithCuda(int *a,int no_elements,int element,int *returned_index)
{
    int *dev_a = 0;
    int *dev_element;
    int *dev_index;

    int h_index=-999;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_a, no_elements * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_index,sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    cudaStatus = cudaMalloc((void**)&dev_element,sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, no_elements * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_element,&element,sizeof(int),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_index,&h_index,sizeof(int),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    // Launch a kernel on the GPU with one thread for each element.
    searchKernel<<<3, 1024>>>(dev_a, dev_element,dev_index);

    // cudaThreadSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaThreadSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(&h_index, dev_index,sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	if(h_index!=-999){
	    printf("\nindex is:%d",h_index);
	    printf("\nindexed element is:%d",a[h_index]);	
	    returned_index[0]=h_index;
	}
	else{
	    printf("index is not found");	
	}

Error:
    cudaFree(dev_a);
    cudaFree(dev_element); 
    cudaFree(dev_index); 

    return cudaStatus;
}

int main()
{
    int arraySize =2079744;
    int i;
    int index=-999;
    int *a=myloadFile(arraySize,"numberslist.txt");
    cudaError_t cudaStatus = searchWithCuda(a,arraySize,1964034,&index);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	
    free(a);
    cudaStatus = cudaThreadExit();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaThreadExit failed!");
        return 1;
    } 
	printf("\nindex:%d\n",index);

    return 0;
}


