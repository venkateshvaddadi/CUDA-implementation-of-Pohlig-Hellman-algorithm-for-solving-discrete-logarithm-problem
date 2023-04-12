// C program to generate random numbers 
#include <stdio.h> 
#include <stdlib.h> 

// Driver program 
int main(void) 
{ 
	FILE *f ;
	f=fopen("input2.txt","w");
	int i;
	for(i = 0; i<100000; i++) 
		fprintf(f," %d ", rand()%1000); 
	return 0; 
} 

