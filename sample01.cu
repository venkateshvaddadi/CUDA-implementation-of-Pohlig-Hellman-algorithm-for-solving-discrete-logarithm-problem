/***
Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
IN THE SOFTWARE.
***/
#include "xmp.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <cuda_runtime_api.h>
 #include<stdint.h>
#include<iostream>
#include <cinttypes>
#define XMP_CHECK_ERROR(fun) \
{                             \
  xmpError_t error=fun;     \
  if(error!=xmpErrorSuccess){ \
    if(error==xmpErrorCuda)   \
      printf("CUDA Error %s, %s:%d\n",cudaGetErrorString(cudaGetLastError()),__FILE__,__LINE__); \
    else  \
      printf("XMP Error %s, %s:%d\n",xmpGetErrorString(error),__FILE__,__LINE__); \
    exit(EXIT_FAILURE); \
  } \
}

using namespace std;
	std::ofstream f;

std::string getText(string filename)
{
	std::string output="";
	std::ifstream file("laststring.txt");
	std::string factor="";
	while (std::getline(file, factor))
	{
		output+=factor+"\n";
 	}	
	//std::cout<<output;
	return output;
}

uint32_t* makeLimbs(std::string input)
{
	uint64_t high = 0, low = 0, tmp;
	for(int i=0;i<input.length();i++)
	{
	    char c=input[i];
	    high *= 10;
	    tmp = low * 10;
	    if(tmp / 10 != low)
	    {
		high += ((low >> 32) * 10 + ((low & 0xf) * 10 >> 32)) >> 32;
	    }
	    low = tmp;
	    tmp = low + c - '0';
	    high += tmp < low;
	    low = tmp;
	}

	uint32_t *number=(uint32_t *)malloc(4*sizeof(uint32_t));
	number[0]=low%4294967296;
	number[1]=low/4294967296;
	number[2]=high%4294967296;
	number[3]=high/4294967296;
	return number;


}
void load_ordered_array(uint32_t *numb1,int no_words,int no_integers,std::string arrayname,int iteration_no)
{
	//f<<""\n"+arrayname+"=[";
	uint32_t value=no_integers*iteration_no;
	printf("iteration no:%d",iteration_no);
	for(int i=0;i<no_integers;i++)
	{
		numb1[4*i]=value+i;
		numb1[4*i+1]=0;
		numb1[4*i+2]=0;
		numb1[4*i+3]=0;

		/*	printf("\n");
		  	printf("%" PRIu32 ":\t",numb1[4*i]);
		  	printf("%" PRIu32 ":\t",numb1[4*i+1]);
		  	printf("%" PRIu32 ":\t",numb1[4*i+2]);
		  	printf("%" PRIu32 ":\t\n",numb1[4*i+3]);
		*/
		//f<<"" "<<numb1[4*i]<<",0,0,0,";
	}
 	//f<<""]\n";
}

void load_gen_array(uint32_t *numb1,int no_words,int no_integers,std::string arrayname)
{
	//f<<""\n"+arrayname+"=[";
	int no_iterations=no_integers*no_words;
	for(int i=0;i<no_iterations ;i++)
	{
		numb1[i]=rand()%4294967296;
		//f<<"" "<<numb1[i]<<",";
	}
 	//f<<""]\n";
}
void load_odd_array(uint32_t *numb1,int no_words,int no_integers,std::string arrayname)
{
	//f<<""\n"+arrayname+"=[";
	int no_iterations=no_integers*no_words;
	for(int i=0;i<no_iterations ;i++)
	{
		numb1[i]=rand()%4294967296;
		//f<<"" "<<numb1[i]<<",";
		if(i%no_words==0&&(numb1[i]%2==0)){
			numb1[i]=numb1[i]+1;
		}
		
		
	}
 	//f<<""]\n";
}
void write_array_in_file(uint32_t *numb1,int no_words,int no_integers,std::string arrayname)
{
	int no_iterations=no_words*no_integers; 
 	//f<<""\n"+arrayname+"=[";
  	for(int i=0;i<no_iterations;i++){
  		//f<<"" "<<numb1[i]<<",";
	}
	//f<<""]\n";
}
__global__ void searchKernel(uint32_t *a, uint32_t *search_element,int *d_index,int *iteration_no){
	int index;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int low=677*tid;
	int high=677*(tid+1)-1;
	uint32_t h0,h1,h2,h3;
	h0=search_element[0];
	h1=search_element[1];
	h2=search_element[2];
	h3=search_element[3];
/*	if(tid==329){
		printf("\niteration_no:%d\n",iteration_no[0]);
	  	printf("\n %d \t" ,low+iteration_no[0]*2079744);
	  	printf("\n %d \t\n" ,high+iteration_no[0]*2079744);
	  	printf("%" PRIu32 ":\t",h0);
	  	printf("%" PRIu32 ":\t",h1);
	  	printf("%" PRIu32 ":\t",h2);
	  	printf("%" PRIu32 ":\t\n",h3);
		
		printf("%d" ,668777>low)	;
		printf("%d",668777<high);	
*/

		for(int i=low;i<=high;i=i+1){


/*		  	printf("\n%d   %d \t" ,i,iteration_no[0]*2079744+i);
		  	printf("%" PRIu32 ":\t",a[4*i]);
		  	printf("%" PRIu32 ":\t",a[4*i+1]);
		  	printf("%" PRIu32 ":\t",a[4*i+2]);
		  	printf("%" PRIu32 ":\t\n",a[4*i+3]);
*/
			if(h0==a[4*i]){
				if(h1==a[4*i+1]){
					if(h2=a[4*i+2]){
						if(h3==a[4*i+3]){
							printf("\nindex found at %d\n",i);
 							d_index[0]=iteration_no[0]*2079744+i;
							break;						
						}					
					}				
				}
			
			}


	 	}


//	}


}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t searchWithCuda(uint32_t *a,int no_elements,uint32_t *element,int *returned_index,int iteration_no)
{
    uint32_t *dev_a;;
    uint32_t *dev_element;
	
    int *dev_index;
		
    int h_index=-999;
    
    int *dev_iteration_no;
			
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_a, no_elements *4* sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_index,sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

 cudaStatus = cudaMalloc((void**)&dev_iteration_no,sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    cudaStatus = cudaMalloc((void**)&dev_element,4*sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, no_elements * 4*sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_element,element,4*sizeof(uint32_t),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_index,&h_index,sizeof(int),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_iteration_no,&iteration_no,sizeof(int),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    // Launch a kernel on the GPU with one thread for each element.
    searchKernel<<<3, 1024>>>(dev_a, dev_element,dev_index,dev_iteration_no);

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
		printf("\nindex is:%d\n",h_index);
	  	printf("%" PRIu32 ":\t",a[h_index%2079744]);
		printf("%" PRIu32 ":\t",a[h_index%2079744+1]);
		printf("%" PRIu32 ":\t",a[h_index%2079744+2]);
		printf("%" PRIu32 ":\t\n",a[h_index%2079744+3]);
		returned_index[0]=h_index;
	}
	else{
	    printf("\nindex is not found\n");	
	}

Error:
    cudaFree(dev_a);
    cudaFree(dev_element); 
    cudaFree(dev_index); 

    return cudaStatus;
}
void display_number(uint32_t *number,int no_words)
{
	printf("\n");
	for(int i=0;i<4;i++){
	  	printf("%" PRIu32 ":\t",number[i]);
	}
	printf("\n");
}
void set_for_test(uint32_t *resb){
	int test_index=2079713;	
	resb[test_index]=123456;
	resb[test_index+1]=789123;
	resb[test_index+2]=456789;
	resb[test_index+3]=234567;
}
uint32_t *get_search_element(){

	uint32_t *search_element=(uint32_t *)malloc(4*sizeof(uint32_t));
	search_element[0]=123456;
	search_element[1]=789123;
	search_element[2]=456789;
	search_element[3]=234567;
	return search_element;
}

int main() {

	f.open ("testoutput.py");

	uint32_t i,w;
	int bits=128;
	int no_integers=2079744;

	uint32_t *numb1,*numb2,*numb3,*resb;	//these are integer variable at the CPU
	uint32_t *search_element;
 	size_t bytes=bits/8;
	uint32_t no_limbs=bytes/sizeof(uint32_t);
	uint32_t limbs2=2*(bytes/sizeof(uint32_t));

	time_t my_time = time(NULL); 
	std::string search_string="239157056073798794349891607716068883257";
 	search_element=makeLimbs(search_string);

  // creating integers at the host
	std::string input1="202572898398210545140461686546660877937";
	std::string input3="305528117913166920104874918136902167483";
 	numb1=makeLimbs((std::string)input1);
	numb2=(uint32_t*)malloc(no_integers*bytes);
	numb3=makeLimbs((std::string)input3);
	resb= (uint32_t*)malloc(no_integers*bytes);

	xmpIntegers_t num1, num2,num3, res;	//these are integer variable at the GPU
	xmpHandle_t handle;
	cudaSetDevice(0);
	//allocate handle
	XMP_CHECK_ERROR(xmpHandleCreate(&handle));
	//allocate integers
	XMP_CHECK_ERROR(xmpIntegersCreate(handle,&num1,bits,1));
	XMP_CHECK_ERROR(xmpIntegersCreate(handle,&num2,bits,no_integers));
	XMP_CHECK_ERROR(xmpIntegersCreate(handle,&num3,bits,1));
	XMP_CHECK_ERROR(xmpIntegersCreate(handle,&res,bits,no_integers));
	
	int no_iterations=no_limbs;

 	//load_odd_array(numb1,4,1,"a");//4 words means 128 bit number

 	//load_odd_array(numb3,4,1,"c");//4 words means 128 bit number

	display_number(numb1,4);
	display_number(numb3,4);
  	
	int iteration_no=0;
	while(iteration_no<=600){
		
		load_ordered_array(numb2,4,no_integers,"b",iteration_no);
		XMP_CHECK_ERROR(xmpIntegersImport(handle,num1,4,-1, sizeof(uint32_t),0,0,numb1,1));//export to gpu
		XMP_CHECK_ERROR(xmpIntegersImport(handle,num2,4,-1, sizeof(uint32_t),0,0,numb2,no_integers));//export to gpu
		XMP_CHECK_ERROR(xmpIntegersImport(handle,num3,4,-1, sizeof(uint32_t),0,0,numb3,1));//export to gpu
		XMP_CHECK_ERROR(xmpIntegersPowm(handle,res,num1,num2,num3,no_integers));//calling power function
		XMP_CHECK_ERROR(xmpIntegersExport(handle,resb,&no_limbs,-1,sizeof(uint32_t),0,0,res,no_integers));//export to host
	 


		int index=-99;;
		cudaError_t cudaStatus = searchWithCuda(resb,no_integers,search_element,&index,iteration_no);
					// searchWithCuda( uint32_t element,int *returned_index)
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}
		if(index!=-99){
			printf("\nindex is:%d\n",index);
		  	printf("%" PRIu32 ":\t",resb[index%2079744]);
			printf("%" PRIu32 ":\t",resb[index%2079744+1]);
			printf("%" PRIu32 ":\t",resb[index%2079744+2]);
			printf("%" PRIu32 ":\t\n",resb[index%2079744+3]);
			break;
		}
		iteration_no=iteration_no+1;
 
	}	

/*
	no_iterations=no_limbs*no_integers; 
 
	//f<<""\nresult=[";
  	for(i=0;i<no_iterations;i++){
  		//f<<"" "<<resb[i]<<",";
	}
	std::cout<<"]\n";
	//f<<""]\n";
*/ 
	//write_array_in_file(resb,4,no_integers,"result");
	std::string need_append=getText("laststring.txt");
	//std::cout<<need_append;
  	//f<<"need_append;
   //free integers
	XMP_CHECK_ERROR(xmpIntegersDestroy(handle,num1));
	XMP_CHECK_ERROR(xmpIntegersDestroy(handle,num3));
	XMP_CHECK_ERROR(xmpIntegersDestroy(handle,res));
	XMP_CHECK_ERROR(xmpIntegersDestroy(handle,num2));

  //free handle
	XMP_CHECK_ERROR(xmpHandleDestroy(handle));
	free(numb1);
	free(resb);
	free(numb2);
	free(numb3);
	time_t my_time1 = time(NULL);
 	printf("%s", ctime(&my_time));
	printf("%s", ctime(&my_time1)); 
	printf("\n\n\nsample01 executed sucessfully\n");
	return 0;
}
