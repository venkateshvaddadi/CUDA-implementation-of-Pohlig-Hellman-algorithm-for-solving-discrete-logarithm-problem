#include<iostream>
#include <bits/stdc++.h>
#include <string>
using namespace std;
int main(int argc,char *argv[])
{
	time_t start_time = time(NULL); 
   	std::string str = "python testoutput.py ";;
   	const char *command = str.c_str(); 
  	system(command); 
	time_t end_time = time(NULL); 
 	printf("%s", ctime(&start_time));
	printf("%s", ctime(&end_time)); 

 
}
