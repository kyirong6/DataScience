import sys
import xml.etree.ElementTree as ET
import pandas as pd

#f1 = sys.argv[1]

tree = ET.parse("walk1.gpx")
root = tree.getroot()
dfcols = ["lat", "lon"]
df = pd.DataFrame(columns=dfcols)
for child in root:
    print(child.tag, child.attrib)

