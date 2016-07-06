import sys


act_separator = "#"


def getId(date, name, location):
    return date + act_separator + location + act_separator + name

def obtainActuation(id):
    elements = id.split(act_separator)
    return Actuation(elements[0],elements[1],elements[2])

class Actuation:
    def __init__(self, date, location, name):
        self._name = name
        self._participations = {}
        self._participants = []
        self._location = location
        self._id = getId(date, name, location)

    def addParticipant(self,crew_name,results):
        if not crew_name in self._participants:
            self._participants.append(crew_name)
            self._participations[crew_name]=results.split(" ")

    def getParticipation(self,crew_name):
        return self._participations[crew_name]

    def getParticipants(self):
        return self._participants