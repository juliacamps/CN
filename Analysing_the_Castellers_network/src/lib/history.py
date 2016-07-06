import sys
from temporada import Temporada

class History:
    def __init__(self):
        self._seasons = {}

    def get_season(self,season_id):
        return self._seasons[season_id]

    def build(self,seasons_content,seasons_ids):
        self._seasons = {seasons_ids[i]: Temporada(seasons_ids[i]).build(seasons_content[seasons_ids[i]]) for i in range(0, len(seasons_ids), 1)}

    def getSeason(self,season):
        return self._seasons[season]