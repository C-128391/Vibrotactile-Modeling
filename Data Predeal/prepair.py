import xml.sax
import os
import pandas as pd
import numpy as np

#  Convert XML file to Excel file，downsample and normalize the original data.

class MyContentHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.values = []
        self.in_value = False

    def startElement(self, name, attrs):
        if name == "value":
            self.in_value = True

    def characters(self, content):
        if self.in_value:
            try:
                value = float(content)
                self.values.append(value)
            except ValueError:
                pass

    def endElement(self, name):
        if name == "value":
            self.in_value = False

folder_path = r'original data/Speed' #The folder containing the original velocity information of the dataset (.xml files).
file_names = os.listdir(folder_path)

for i in file_names:
    xml_file = r"original data/Speed/%s" %i
    parser = xml.sax.make_parser()
    handler = MyContentHandler()
    parser.setContentHandler(handler)
    parser.parse(xml_file)

    speed_values = handler.values
    speed_values = speed_values[:400000][::10]
    print(len(speed_values))

    name = i.rsplit('.', 1)[0]
    name = name + '.xlsx'
    excel_file = r"original data/data_deal_test/original_force/%s"%name #The folder for storing preprocessed information.

    df = pd.DataFrame({'File Name': speed_values})

    df.to_excel(excel_file, index=False)





