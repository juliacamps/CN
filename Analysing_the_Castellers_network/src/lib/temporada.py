import sys
from actuacio import getId, Actuation, obtainActuation
from colla import Colla

class Temporada:

    def __init__(self, id):
        self._name = id
        self._actuations = {}
        self._crews = {}

    def build(self,data_content):
        self._actuations = {}
        # print(type(data_content))
        # print(data_content)
        # data_content.items()
        for item in data_content:
            # instance = item[1]
            actuation = self._getActuation(getId(item.date,item.loc,item.event))
            actuation.addParticipant(item.crew,item.results)
            crew = self._getCrew(item.crew)
            crew.addResults(item.results)
        for crew in self._crews.items():
            crew[1].setLevel()
            crew[1].setUnsafeness()
        return self

    def _getCrew(self,crew_name):
        if crew_name in self._crews:
            ret = self._crews[crew_name]
        else:
            ret = Colla(crew_name)
            self._crews[crew_name] = ret
        return ret

    def _getActuation(self,act_id):
        if act_id in self._actuations:
            res = self._actuations[act_id]
        else:
            res = obtainActuation(act_id)
            self._actuations[act_id]=res
        return res

    def getCrews(self):
        return self._crews

    def getActuations(self):
        return self._actuations

