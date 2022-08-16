import random


class ModifyList:
    def __init__(self, actual_list, mod_factor, percentage_of_total_anomaly):
        self.selects = []
        self.create_random_selections(len(actual_list), percentage_of_total_anomaly)
        self.modified_list = []
        self.create_modified_list(actual_list, mod_factor)

    def create_random_selections(self, length, percentage_of_total_anomaly):
        sample_size = int(length*percentage_of_total_anomaly/100)
        self.selects = random.sample(range(0, length), sample_size)
        self.selects.sort()

    def create_modified_list(self, actual_list, mod_factor):
        self.modified_list = actual_list
        for index in self.selects:
            self.modified_list[index] *= mod_factor




