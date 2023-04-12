#include<iostream>
#include <cinttypes>
using namespace std;

uint32_t* makeLimbs(string input)
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
void display_number(uint32_t *number,int no_words)
{
	printf("\n");
	for(int i=0;i<4;i++){
	  	printf("%" PRIu32 ":\t",number[i]);
	}
	printf("\n");
}
int main()
{
 
	uint32_t *number1=makeLimbs("230637460016956417767082848376839080233");
	uint32_t *number2=makeLimbs("333232988166217712727077362427791161391");
	display_number(number1,4);
 	display_number(number2,4);

}
