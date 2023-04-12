#include<iostream>
#include <bits/stdc++.h>
#include <string>
using namespace std;
void makesubproblems(string input1,string input2,string input3,string *g,string *y,int *no_problems){
	printf("\nyou are in make_factors method:\n");
 	std::string str = "python2 test.py "+input1+" "+input2+" "+input3;
  	const char *command = str.c_str(); 
	system(command); 
	std::ifstream file("your_file.txt");
	std::string factor;
	std::string len;
	std::getline(file, len);
	int no_factors=std::atoi(len.c_str());	
	no_problems[0]=no_factors;
	std::cout<<"No of subproblems are:"<<no_factors<<std::endl;
	int i=0;
 	int problem_no=0;
	while (std::getline(file, factor))
	{
 		int need_to_break=factor.find(",");
 		g[problem_no]=factor.substr(0,need_to_break);
		y[problem_no]=factor.substr(need_to_break+1,factor.length());
		//std::cout<<g[problem_no]<<std::endl;
		//std::cout<<y[problem_no]<<std::endl;
 		problem_no=problem_no+1;
	}

}
int main(int argc,char *argv[])
{
	string input1=std::string(argv[1]);
	string input2=std::string(argv[2]);
	string input3=std::string(argv[3]);
	std::string g[5];
	std::string y[5];
	int no_problems;
	makesubproblems(input1,input2,input3,g,y,&no_problems);
	printf("no of problems are: %d\n",no_problems);
	for(int i=0;i<5;i++){
		std::cout<<g[i]<<" ";
		std::cout<<y[i]<<std::endl;
	}
	
	

}

