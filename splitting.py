#way 1
list1=['x','y','z','a','b','c','d','e','f','g']
find=raw_input("Enter string to be found")
l=list1.index(find)
list1a=[:l]
list1b=[l:]

#way 2
list1 = ['x','y','z','a','b','c','d','e','f','g']
indices = [(0, 4), (5, 9)]
print [list1[s:e+1] for s,e in indices]
