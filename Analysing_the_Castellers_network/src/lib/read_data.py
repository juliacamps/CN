import os
import sys


class Instance:
    def __init__(self,id):
        self.id = id

class Data:
    def __init__(self, separator=" ", ini=1990, fin=2015, path=""):
        self._separator = separator
        self._in_period = ini
        self._fi_period = fin
        if path == "":
            self._raw_content = ""
        else:
            with open(path, 'rb') as ref_file:
                self._raw_content = ref_file.read().decode("utf-8")


    def read_raw(self):
        return self._raw_content

    def read_dataset(self,season_sep):
        data_content = {}
        seasons_data = self._raw_content.split(season_sep)
        count = 0
        for i in range(1,len(seasons_data),1):
            season_data = seasons_data[i]
            data_season = []
            items = season_data.split('\n')[:-1]
            for item in items:
                count += 1
                parts = item.split(self._separator)
                instance = Instance(count)
                instance.date = parts[0]
                instance.loc = parts[1]
                instance.event = parts[2]
                instance.crew = parts[3]
                instance.results = parts[4]
                data_season.append(instance)
            data_content[self._in_period+i-1] = data_season
        return data_content