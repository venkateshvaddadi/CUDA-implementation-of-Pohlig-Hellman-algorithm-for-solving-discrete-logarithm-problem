#include<iostream>
#include<iostream>
#include <bits/stdc++.h>
#include <string>
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
int main()
{
	std::string output="";
	std::ifstream file("laststring.txt");
	std::string factor="";
	while (std::getline(file, factor))
	{
        	output+=factor+"\n";
 	}	
	std::cout<<output;
	std::cout<<getText("laststring.txt");
}

