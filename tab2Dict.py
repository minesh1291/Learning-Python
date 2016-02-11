def makeDict(toDictFile):
	FH=open(toDictFile,"r")
	lines=FH.read().splitlines()
	FH.close()
	lines=map(lambda x: x.split("\t"),lines)
	words=[item for sublist in lines for item in sublist]
	dicti= dict(zip(words[0::2], words[1::2]))
	return dicti

'''
input:
keys  values
k1  v1
k2  v2
k3  v3
'''
