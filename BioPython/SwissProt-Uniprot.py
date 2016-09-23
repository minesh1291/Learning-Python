# SwissProt / Uniprot flat file parsing 
from Bio import SwissProt
for record in SwissProt.parse(open('/path/to/your/uniprot_sprot.dat')):
  pid=record.accessions[0]
  seq=record.sequence
  for feature in record.features:
    print feature
