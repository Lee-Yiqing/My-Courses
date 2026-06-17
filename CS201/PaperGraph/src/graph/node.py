"""Node class for heterogeneous graph."""


class Node:
    __slots__ = ("id", "type", "properties")

    def __init__(self, node_id, node_type, properties=None):
        self.id = node_id
        self.type = node_type  # "paper", "author", "keyword", "blog_post"
        self.properties = properties or {}

    def __repr__(self):
        return f"Node({self.id}, type={self.type})"

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)