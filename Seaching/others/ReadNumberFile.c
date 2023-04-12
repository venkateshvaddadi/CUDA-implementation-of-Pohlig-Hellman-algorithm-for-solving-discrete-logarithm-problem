// C program to generate random numbers 
#include <stdio.h> 
#include <stdlib.h> 

// Driver program
int * myloadFile(int n,char *s){
	FILE *f=fopen(s,"r");
	int i;
	int *a=(int *)malloc(sizeof(int)*1000);
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
		printf("%d\n",a[i]);
 	}
		 


}
 
int main(void) 
{  
	int n=1000;
	int *a=myloadFile(n,"input1.txt");
	int *b=myloadFile(n,"input2.txt");
 	display(a,n)	;
 	display(b,n)	;
			 
	return 0; 
}
