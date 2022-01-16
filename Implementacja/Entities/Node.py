from Implementacja.Entities.Connection import Connection


class Node:
    def __init__(self, parent_conn, split_crit=None, class_label=None):
        self.parent_conn = parent_conn
        self.split_crit = split_crit
        self.class_label = class_label
        self.child_conn = []

    def is_leaf(self):
        return self.class_label is not None

    def attach_child_connection(self, attribute_value):
        self.child_conn.append(Connection(self, attribute_value, None))

    def attach_a_child(self, attribute_value, child):
        conn = Connection(self, attribute_value, child)
        self.child_conn.append(conn)
        child.parent_conn = conn

    def attach_parent_connection(self, connection):
        self.parent_conn = connection

    def __dict__(self):
        if self.is_leaf():
            if self.parent_conn is not None:
                return {
                           "split_value": str(self.parent_conn.attribute_value),
                           "class_label": self.class_label,
                       } | {f"children{i}": child_conn.child_node.to_dict() for i, child_conn in enumerate(self.child_conn)}
            else:
                return {
                           "class_label": self.class_label,
                       } | {f"children{i}": child_conn.child_node.to_dict() for i, child_conn in enumerate(self.child_conn)}
        else:
            if self.parent_conn is not None:
                return {
                           "split_value": str(self.parent_conn.attribute_value),
                           "split_crit": self.split_crit,
                       } | {f"children{i}": child_conn.child_node.to_dict() for i, child_conn in enumerate(self.child_conn)}
            else:
                return {
                           "split_crit": self.split_crit,
                       } | {f"children{i}": child_conn.child_node.to_dict() for i, child_conn in enumerate(self.child_conn)}

    def to_dict(self):
        return self.__dict__()
