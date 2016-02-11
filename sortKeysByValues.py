def sortVal(x):
	import operator
	sorted_x = sorted(x.items(), key=operator.itemgetter(1),reverse=True)
	return sorted_x
