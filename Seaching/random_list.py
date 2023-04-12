import math
import random;
a=[]
for i in range(2079744):
	a.append(i);
print(len(a));
random.shuffle(a);


f= open("numberslist.txt","w+");
for i in a:
	f.write("%d\n" % (i))
