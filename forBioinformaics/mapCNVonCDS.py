#!/usr/bin/python
import csv, sys, multiprocessing, threading
from multiprocessing import Pool, Value, Lock


class Counter(object):
    def __init__(self, initval=0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value


counter = Counter(0)


if len(sys.argv) != 4:
	print >> sys.stderr, "USAGE: "+sys.argv[0] +" gff_File cnv_File OutGff"
	sys.exit(1)

OutGff=sys.argv[3]

gff_File=sys.argv[1]#"Homo_gtf.fil.or"
gff_FH=open(gff_File,'r')
temp=csv.reader(gff_FH,delimiter='\t')
GFF_cont = [line for line in temp]
GFF_count=len(GFF_cont)

for i in range(0, len(GFF_cont)-1):
	GFF_cont[i][3]= int(GFF_cont[i][3])
	GFF_cont[i][4]= int(GFF_cont[i][4])
	

cnv_File=sys.argv[2]#"cnv_out3.txt"
cnv_FH=open(cnv_File,'r')
temp=csv.reader(cnv_FH,delimiter='\t')
CNV_cont = [line for line in temp]
CNV_count=len(CNV_cont)

for i in range(0, len(CNV_cont)-1):
	CNV_cont[i][1]= int(CNV_cont[i][1])
	CNV_cont[i][2]= int(CNV_cont[i][2])
	


# Create Object of Manager and Queue
manager = multiprocessing.Manager()
q = manager.Queue()
qe = manager.Queue()


def Search(cnv_entry):
	found=0
	counter.increment()
	
	print >> sys.stderr, "\r" + str(counter.value()) + " Out of " + str(CNV_count)
	
	#~ name=threading.current_thread().name
	#~ print >> sys.stderr, name
	
	
	for gff_entry in GFF_cont:
		
		if "chr"+gff_entry[0] != cnv_entry[0]:
			continue
		
		if gff_entry[3]<= cnv_entry[1] and gff_entry[4]>=cnv_entry[1]:
			
			#condition1
			if cnv_entry[2] <= gff_entry[4]:
				q.put('\t'.join([cnv_entry[0],str(cnv_entry[1]),str(cnv_entry[2]),cnv_entry[3],cnv_entry[4],cnv_entry[5],cnv_entry[6],cnv_entry[7],cnv_entry[8],cnv_entry[9]]) + "\t" + 
					'\t'.join([gff_entry[1],gff_entry[2],str(gff_entry[3]),str(gff_entry[4]),gff_entry[6],gff_entry[8]])+ "\tCNV1\t" + cnv_entry[3] + "\n" )
				found=1
				
			#condition2
			else:
				cover=gff_entry[4]-cnv_entry[1]
				#~ q.put('\t'.join(cnv_entry) + "\t" + '\t'.join([gff_entry[1],gff_entry[2],gff_entry[3],gff_entry[4],gff_entry[6],gff_entry[8]])+ "\tCNV2\t" + str(cover) + "\n" )
				q.put('\t'.join([cnv_entry[0],str(cnv_entry[1]),str(cnv_entry[2]),cnv_entry[3],cnv_entry[4],cnv_entry[5],cnv_entry[6],cnv_entry[7],cnv_entry[8],cnv_entry[9]]) + "\t" + 
					'\t'.join([gff_entry[1],gff_entry[2],str(gff_entry[3]),str(gff_entry[4]),gff_entry[6],gff_entry[8]])+ "\tCNV2\t" + str(cover) + "\n" )
				found=1
				
		#condition3
		elif gff_entry[3]<=cnv_entry[2] and gff_entry[4]>= cnv_entry[2] :
			cover=cnv_entry[2]-gff_entry[3]
			#~ q.put('\t'.join(cnv_entry) + "\t" + '\t'.join([gff_entry[1],gff_entry[2],gff_entry[3],gff_entry[4],gff_entry[6],gff_entry[8]])+ "\tCNV3\t" + str(cover) + "\n" )
			q.put('\t'.join([cnv_entry[0],str(cnv_entry[1]),str(cnv_entry[2]),cnv_entry[3],cnv_entry[4],cnv_entry[5],cnv_entry[6],cnv_entry[7],cnv_entry[8],cnv_entry[9]]) + "\t" + 
					'\t'.join([gff_entry[1],gff_entry[2],str(gff_entry[3]),str(gff_entry[4]),gff_entry[6],gff_entry[8]])+ "\tCNV3\t" + str(cover) + "\n" )
			found=1
					
		#condition4
		elif gff_entry[3]>=cnv_entry[1] and gff_entry[4]<=cnv_entry[2] :
			cover=gff_entry[4]-gff_entry[3]
			#~ q.put('\t'.join(cnv_entry) + "\t" + '\t'.join([gff_entry[1],gff_entry[2],gff_entry[3],gff_entry[4],gff_entry[6],gff_entry[8]])+ "\tCNV4\t" + str(cover) + "\n" )
			q.put('\t'.join([cnv_entry[0],str(cnv_entry[1]),str(cnv_entry[2]),cnv_entry[3],cnv_entry[4],cnv_entry[5],cnv_entry[6],cnv_entry[7],cnv_entry[8],cnv_entry[9]]) + "\t" + 
					'\t'.join([gff_entry[1],gff_entry[2],str(gff_entry[3]),str(gff_entry[4]),gff_entry[6],gff_entry[8]])+ "\tCNV4\t" + str(cover) + "\n" )
			found=1
		#
		if gff_entry[4] < cnv_entry[1]:
			break
	
	if found == 0 :
		qe.put('\t'.join([cnv_entry[0],str(cnv_entry[1]),str(cnv_entry[2]),cnv_entry[3],cnv_entry[4],cnv_entry[5],cnv_entry[6],cnv_entry[7],cnv_entry[8],cnv_entry[9]])+"\n")
	
	
	return


##
#	Main Function
##

def main():
	
	num_cores = multiprocessing.cpu_count()
	ProcNum = 5*num_cores
# Create Object of Pool
	P = Pool(processes=ProcNum)

# Make Genome Sequence List 
	# Prepare Argument for Pool.map
	record_list=[record for record in CNV_cont]

# Map pool of workers
	P.map(Search,record_list) 

# Display Ongoing Process
	print >> sys.stderr,"Writing Output to File : " + OutGff

# Write queue to output file
	with open(OutGff, 'w') as fp:
		while not q.empty():
			item = q.get()
			fp.write(item)
	
	with open(OutGff+'.dump', 'w') as fp:
		while not qe.empty():
			item = qe.get()
			fp.write(item)
	
	return


##
#	Main Area
##

if __name__ == '__main__':
	
	main()
	
'''
seq -------------------

cds       -------      
cnv1         ---       

cds       -------      
cnv2          -----    

cds       -------      
cnv3    ----           

cds       -------      
cnv4     -----------    
'''
