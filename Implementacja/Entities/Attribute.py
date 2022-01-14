
class Attribute:
    def __init__(self, name):
        self.name = name
        self.values = []

    def add_value(self, value):
        self.values.append(value)


