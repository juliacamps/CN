import sys
from castell import getCastells
from evaluator import crew_level, castell_level, getTrialLevel, evaluateFall, isFall, evaluated


class Colla:
    def __init__(self,name):
        self._name = name
        self._growth_index = 0
        self._level = 0
        self._castells = []
        self._season = 0
        self._unsafeness = 0

    def getLevel(self):
        return self._level

    def setLevel(self):
        level = max([crew_level(x) for x in self._castells])
        self._level = level


    def addResults(self,results):
        castells = getCastells(results)
        self._castells += castells


    def getGrowth(self):
        return self._growth_index

    def calculateGrowth(self,previous):
        self._growth_index = 0.5 * (previous.getGrowth() + (self.getLevel()-previous.getLevel()))
        return self.getGrowth()

    def setSeason(self,season):
        self._season = season

    def getSeason(self):
        return self._season

    def getTopCastells(self,num):
        candidates = (list(set([castell_level(x) for x in self._castells])))
        candidates.sort()
        return candidates[-num:]

    def getTopTrials(self, num):
        candidates = (list(set([getTrialLevel(x) for x in self._castells])))
        candidates.sort()
        return candidates[-num:]

    def getUnsafeness(self):
        return self._unsafeness

    def calculateUnsafeness(self,previous):
        self._unsafeness = 0.5 * (previous.getUnsafeness() + self.getUnsafeness())
        return self.getUnsafeness()

    def setUnsafeness(self):
        self._unsafeness = sum([evaluateFall(getTrialLevel(x),self.getLevel()) for x in self._castells if isFall(x)])/float(len([x for x in self._castells if evaluated(x)])+1)

