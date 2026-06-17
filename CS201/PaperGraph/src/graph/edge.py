"""Edge class for heterogeneous graph."""


class Edge:
    __slots__ = ("source", "target", "type", "weight", "properties")

    def __init__(self, source, target, edge_type, weight=1.0, properties=None):
        self.source = source    # node_id
        self.target = target    # node_id
        self.type = edge_type   # "cite", "author_of", "topic", "method", "sequence"
        self.weight = weight
        self.properties = properties or {}

    def __repr__(self):
        return f"Edge({self.source}->{self.target}, type={self.type}, w={self.weight})"

    def __eq__(self, other):
        return (isinstance(other, Edge)
                and self.source == other.source
                and self.target == other.target
                and self.type == other.type)

    def __hash__(self):
        return hash((self.source, self.target, self.type))