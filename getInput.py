import pickle
import numpy as np
test1 = []
each1 = []
trainf1 = []

test2 = []
each2 = []
trainf2 = []

test3 = []
each3 = []
trainf3 = []
# we need a list of 8 vectors, each of length 13 for 8 RNN input units

for a in range(1,99):

	ds = pickle.load(open("/home/dhanya/Documents/Machine Learning/Neural Networks/RNN/MFCC/fours/f"+str(a)+"_mfcc.p","rb"))
	inputs = ds[:]
	for i in inputs:
		test1.append(i)

for a in range(1,64):

	ds = pickle.load(open("/home/dhanya/Documents/Machine Learning/Neural Networks/RNN/MFCC/outs/w"+str(a)+"_mfcc.p","rb"))
	inputs = ds[:]
	for i in inputs:
		test2.append(i)

for a in range(1,27):

	ds = pickle.load(open("/home/dhanya/Documents/Machine Learning/Neural Networks/RNN/MFCC/sixes/s"+str(a)+"_mfcc.p","rb"))
	inputs = ds[:]
	for i in inputs:
		test3.append(i)



# has all fours
#print(test1)

for i in range(0,len(test1),8):
	each1.append(test1[i:i+8])

trainf1 = each1[:int(0.8*len(each1))]
testf1 = each1[int(0.8*len(each1)+1):]


#outs
for i in range(0,len(test2),8):
	each2.append(test2[i:i+8])

trainf2 = each2[:int(0.8*len(each2))]
testf2 = each2[int(0.8*len(each2)+1):]
print(len(testf2[-1]))
print(len(each2))
print(len(test2))

#sixes
for i in range(0,len(test3),8):
	each3.append(test3[i:i+8])

trainf3 = each3[:int(0.8*len(each3))]
testf3 = each3[int(0.8*len(each3)+1):]

'''for eachEight in trainf:

  for t in range(len(eachEight)):	# 8
    for i in range(len(eachEight[t])):
      temp = np.zeros(shape = (13,1))
      temp[i] = eachEight[t][i]
      print(temp[0])'''
#print(trainf[0],len(trainf[0]),len(trainf[0][0]))	# right
	
	
