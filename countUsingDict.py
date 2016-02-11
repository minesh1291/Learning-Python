def add2dict(d,k):
	if k in d:
		d[k]+=1
	else:
		d[k]=1
	return d
def sortVal(x):
	import operator
	sorted_x = sorted(x.items(), key=operator.itemgetter(1),reverse=True)
	return sorted_x

#species_list=[<input goes here>]
species_cnt={}

for species in species_list:
  species_cnt=add2dict(species_cnt,species)

for k in sortVal(species_cnt):
	print k[0],"\t",k[1]
print
