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
                       } | {f"child{i}": child_conn.child_node.to_dict() for i, child_conn in
                            enumerate(self.child_conn)}
            else:
                return {
                           "class_label": self.class_label,
                       } | {f"child{i}": child_conn.child_node.to_dict() for i, child_conn in
                            enumerate(self.child_conn)}
        else:
            if self.parent_conn is not None:
                return {
                           "split_value": str(self.parent_conn.attribute_value),
                           "split_crit": self.split_crit,
                       } | {f"child{i}": child_conn.child_node.to_dict() for i, child_conn in
                            enumerate(self.child_conn)}
            else:
                return {
                           "split_crit": self.split_crit,
                       } | {f"child{i}": child_conn.child_node.to_dict() for i, child_conn in
                            enumerate(self.child_conn)}

    def to_dict(self):
        return self.__dict__()

    def predict(self, sample, continuous_attributes):
        if self.is_leaf():
            return self.class_label
        else:
            val = sample[self.split_crit]
            if self.split_crit not in continuous_attributes:
                for conn in self.child_conn:
                    if conn.attribute_value == val:
                        return conn.child_node.predict(sample, continuous_attributes)
            else:
                for conn in self.child_conn:
                    threshold = conn.attribute_value
                    # more
                    if threshold[0] == ">":
                        value = float(threshold[1:])
                        if val > value:
                            return conn.child_node.predict(sample, continuous_attributes)
                    # less
                    else:
                        value = float(threshold[2:])
                        if val <= value:
                            return conn.child_node.predict(sample, continuous_attributes)

            # if we are here, there was not an exact match found for the sample :(
            # return the the most common class among subtrees
            class_list = []
            self.get_list_of_possible_classes(class_list)
            return max(set(class_list), key=class_list.count)

    def get_list_of_possible_classes(self, class_list=None):
        if class_list is None:
            class_list = []

        for conn in self.child_conn:
            child = conn.child_node
            if child.is_leaf():
                class_list.append(child.class_label)
            else:
                child.get_list_of_possible_classes(class_list)
