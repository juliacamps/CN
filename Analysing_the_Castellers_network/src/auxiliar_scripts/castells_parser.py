import os

raw_path = os.getcwd()+r"\..\other_data\castells.txt"
hist_path = os.getcwd()+r"\..\other_data\castells_parsed.txt"


with open(raw_path, 'rb') as raw_file:
    aux1 = raw_file.read().decode("utf-8")
aux1 = aux1.replace(" ","")
aux1 = aux1.replace("\r", "")
aux1 = aux1.replace("\t", "")
aux1 = aux1.replace("e", "")
aux1 = aux1.replace("\n","#")
with open(hist_path, 'wb') as hist_file:
    hist_file.write(str.encode(aux1))
print("parsing performed: OK")

