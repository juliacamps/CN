import sys
from evaluator import castell_level

def getCastells(results):
    return results.split(" ")


class Castell:
    def __init__(self,name):
        self._name = name
        self._rank = crew_level(name)
