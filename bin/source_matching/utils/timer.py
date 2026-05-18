from datetime import datetime


class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = datetime.now()

    def get_time_delta_string(self):
        return datetime.now() - self.start_time

    def __del__(self):
        print(f'{self.name}: {self.get_time_delta_string()}')
