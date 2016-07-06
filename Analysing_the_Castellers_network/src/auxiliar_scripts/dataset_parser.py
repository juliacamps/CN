import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
import random as rd


import unicodedata
def normalize_text(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn' and (c.isalnum() or c.isspace() or c=='/'))
# plt.close("all")
in_period = 1990
fi_period = 2015
raw_path = os.getcwd()+r"\..\raw_data\raw_"
ref_path = os.getcwd()+r"\..\data\temp_"
hist_path = os.getcwd()+r"\..\data\hist_"+str(in_period)+"_"+str(fi_period)+".txt"
colles_excluded = ["Arreplegats de la Zona Universitaria","Emboirats de la Universitat de Vic","Ganapies de la UAB",
                   "Llunatics UPC Vilanova","Xoriguers de la UdG","Engrescats de URL","Penjats del Campus de Manresa",
                   "Marracos de la Universitat de Lleida","Trempats de la UPF","Grillats del Campus del Baix Llobregat",
                   "Pataquers de la URV","Bergants del Campus de Terrassa","Passerells del TCM"]
colla_index = 4
raw_content = bytearray()
with open(hist_path, 'wb') as hist_file:
    for i in range(in_period,fi_period+1):
        with open(raw_path+str(i)+".txt", 'rb') as raw_file:
            raw_content+=(str.encode("NEW_TEMP\r\n"))
            aux_content = raw_file.read()
            st_index = aux_content.find((b'\n'), 1)+1
            raw_content +=aux_content[st_index:]
    str_hist = raw_content.decode("utf-8")
    str_hists = str_hist.split("NEW_TEMP\r\n")
    it = 0
    hists_ref = []
    for temp_hist in str_hists:
        year = str(in_period+it)
        lines_hist = temp_hist.split("\r\n")
        lines_ref = []
        for j in range(0,len(lines_hist)-1,2):
            line_hist = lines_hist[j]+"\t"+lines_hist[j+1]
            parts_line = line_hist.split("\t")
            for i in range(0,len(parts_line),1):
                parentesis_index_1 = parts_line[i].find("(")
                if parentesis_index_1 > -1:
                    parentesis_index_2 =  parts_line[i].find(')')
                    if parentesis_index_2 > -1 and parentesis_index_2 > parentesis_index_1:
                        parts_line[i] = parts_line[i][:parentesis_index_1-1]
            parts_line_ref = []
            for part_line in parts_line:
                part_line_ref = normalize_text(part_line)
                parts_line_ref.append(part_line_ref)
            if not (parts_line_ref[colla_index] in colles_excluded):
                parts_ref = "#".join(parts_line_ref[1:])
                lines_ref.append(parts_ref)
        hist_ref = "\n".join(lines_ref)
        hists_ref.append(hist_ref)
        it += 1
    final_hists = "\nTEMP_INI\n".join(hists_ref).replace("'","")
    hist_file.write(str.encode(final_hists))
print("parsing performed: OK")
