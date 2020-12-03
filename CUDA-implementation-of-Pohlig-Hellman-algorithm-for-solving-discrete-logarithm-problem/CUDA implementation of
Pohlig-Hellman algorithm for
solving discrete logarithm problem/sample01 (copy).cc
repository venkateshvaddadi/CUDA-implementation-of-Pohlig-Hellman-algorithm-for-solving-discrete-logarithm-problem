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

std::string getText(string filename)
{
	std::string output="";
	std::ifstream file("laststring.txt");
	std::string factor="";
	while (std::getline(file, factor))
	{
		output+=factor+"\n";
 	}	
	std::cout<<output;
	return output;
}


int main() {
	std::ofstream f;
	f.open ("testoutput.py");
	uint32_t i,w;
	int bits=128;
	int no_integers=1048576;

	xmpIntegers_t num1, num2,num3, res;	//these are integer variable at the GPU
	uint32_t *numb1,*numb2,*numb3,*resb;	//these are integer variable at the CPU
 
	size_t bytes=bits/8;
	uint32_t no_limbs=bytes/sizeof(uint32_t);
	uint32_t limbs2=2*(bytes/sizeof(uint32_t));



  // creating integers at the host

	numb1=(uint32_t*)malloc(bytes);
	numb2=(uint32_t*)malloc(no_integers*bytes);
	numb3=(uint32_t*)malloc( bytes);
	resb= (uint32_t*)malloc(no_integers*bytes);

	xmpHandle_t handle;
	cudaSetDevice(0);

	//allocate handle
	XMP_CHECK_ERROR(xmpHandleCreate(&handle));
	//allocate integers
	XMP_CHECK_ERROR(xmpIntegersCreate(handle,&num1,bits,1));
	XMP_CHECK_ERROR(xmpIntegersCreate(handle,&num2,bits,no_integers));
	XMP_CHECK_ERROR(xmpIntegersCreate(handle,&num3,bits,1));
	XMP_CHECK_ERROR(xmpIntegersCreate(handle,&res,bits,no_integers));
	
	int no_iterations=no_limbs;;

	std::cout<<"\na=[";
 	f<<"\na=[";
	for(int i=0;i<no_iterations ;i++)
	{
		numb1[i]=rand()%4294967296;
		std::cout<<" "<<numb1[i]<<",";		
		f<<" "<<numb1[i]<<",";
	}
	std::cout<<"]\n";
	f<<"]\n";
 
	no_iterations=	no_iterations=no_limbs*no_integers;;
 	std::cout<<"\nb=[";
	f<<"\nb=[";
	for(int i=0;i<no_iterations;i++)
	{
		if(i%4==0){
			//printf("\n");	
		}
		numb2[i]=rand()%4294967296;
		std::cout<<" "<<numb2[i]<<",";	 
		f<<" "<<numb2[i]<<",";
	}
	std::cout<<"]\n";	
	f<<"]\n";
	no_iterations=no_limbs;//*no_integers;;
	std::cout<<"\nc=[";
	f<<"\nc=[";	

	for(int i=0;i<(no_iterations);i++)
	{
		numb3[i]=rand()%4294967296;
		if(i%4==0&&(numb3[i]%2==0)){
			numb3[i]=numb3[i]+1;
		}
 		std::cout<<" "<<numb3[i]<<",";	 
		f<<" "<<numb3[i]<<",";

	}
	std::cout<<"]\n";
 	f<<"]\n";
  //import 
	XMP_CHECK_ERROR(xmpIntegersImport(handle,num1,4,-1, sizeof(uint32_t),0,0,numb1,1));
	XMP_CHECK_ERROR(xmpIntegersImport(handle,num2,4,-1, sizeof(uint32_t),0,0,numb2,no_integers));
	XMP_CHECK_ERROR(xmpIntegersImport(handle,num3,4,-1, sizeof(uint32_t),0,0,numb3,1));
  //call powm
	time_t my_time = time(NULL); 
 
	XMP_CHECK_ERROR(xmpIntegersPowm(handle,res,num1,num2,num3,no_integers));

	time_t my_time1 = time(NULL);


  //export
	XMP_CHECK_ERROR(xmpIntegersExport(handle,resb,&no_limbs,-1,sizeof(uint32_t),0,0,res,no_integers));

	
  //use results here 

	no_iterations=no_limbs*no_integers; 
	std::cout<<"\nresult=[";
	f<<"\nresult=[";
 
	for(i=0;i<no_iterations;i++){
		if(i%4==0){
			//printf("\n");	
		}
 		std::cout<<" "<<resb[i]<<",";
 		f<<" "<<resb[i]<<",";
	}
	std::cout<<"]\n";
	f<<"]\n";
 
	std::string need_append=getText("laststring.txt");
	std::cout<<need_append;
  	f<<need_append;
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
	printf("%s", ctime(&my_time));
	printf("%s", ctime(&my_time1)); 
	printf("\n\n\nsample01 executed sucessfully\n");
	return 0;
}
