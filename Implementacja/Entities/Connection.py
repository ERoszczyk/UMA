
class Connection:
    def __init__(self, parent_node, attribute_value, child_node=None):
        self.parent_node = parent_node
        self.attribute_value = attribute_value
        self.child_node = child_node

    def connect_child(self, child_node):
        self.child_node = child_node
