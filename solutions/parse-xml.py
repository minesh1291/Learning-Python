# Import Functions
import urllib2
import xml.etree.ElementTree as ET

# Get online XML file
url="https://cghub.ucsc.edu/cghub/metadata/analysisDetail/a8f16339-4802-440c-81b6-d7a6635e604b"

request=urllib2.Request(url, headers={"Accept" : "application/xml"})
u=urllib2.urlopen(request)

tree=ET.parse(u)
root=tree.getroot()

dict={}
for i in root.iter():
	if i.text!=None:
		dict[i.tag]=i.text.strip()
	else:
		dict[i.tag]=""

for key in sorted(dict.keys(), key=lambda v: v.upper()):
	print key+":"+dict[key]
