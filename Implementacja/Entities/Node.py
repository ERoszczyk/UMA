from Implementacja.Entities.Connection import Connection


class Node:
    def __init__(self, parent_conn, split_crit=None, class_label=None):
        self.parent_conn = parent_conn
        self.split_crit = split_crit
        self.class_label = class_label
        self.child_conn = []

    def is_leaf(self):
        return self.class_label is not None

    def attach_connection(self, attribute_value):
        self.child_conn.append(Connection(self, attribute_value, None))
