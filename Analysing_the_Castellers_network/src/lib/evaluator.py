import sys


castells_all = "2d6#Pd5#9d6#4d7#3d7#3d7a#4d7a#7d7#5d7#7d7a#5d7a#3d7s#9d7#2d7#4d8#Pd6#3d8#7d8#2d8f#Pd7f#5d8#4d8a#3d8a#" \
               "7d8a#5d8a#4d9f#3d9f#9d8#3d8s#2d9fm#Pd8fm#7d9f#5d9f#4d9fa#3d9fa#5d9fa#4d9sf#2d8sf#3d10fm#9d9f#4d10fm#" \
               "2d9sm#Pd9fmp#3d9sf"
castells_list = castells_all.split("#")
castells_table = {castells_list[i]: i + 1 for i in range(0, len(castells_list), 1)}


def crew_level(best_castell):
    res = 0
    if best_castell in castells_table:
        res = castells_table[best_castell]
    return res

def castell_level(castell):
    castell = castell.replace("c", "")
    return crew_level(castell)

def getTrialLevel(castell):
    castell = castell.replace("id","")
    i = castell.find('x')
    if i > 0:
        castell = castell[i+1:]
    return castell_level(castell)

def evaluateFall(fall_val, level):
    return 2**(max(0,fall_val-level))

def isFall(castell):
    castell = castell.replace("cam", "").replace("id", "")
    return castell.find("i")>-1 or castell.find("c")

def evaluated(castell):
    return crew_level(castell)>0#castell.find("i")==-1 and castell.find("d4")==-1 and (castell.find("c")==-1 or castell.find("cam")>-1)

def clean(castell):
    i = castell.find("x")
    if i>-1:
        castell = castell[i:]
    return castell