class Node():
    """A node in the search tree."""
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action


class StackFrontier():
    """A stack-based frontier for depth-first search."""
    def __init__(self):
        self.frontier = []

    def add(self, node):
        """Add a node to the frontier."""
        self.frontier.append(node)

    def contains_state(self, state):
        """Check if a state is already in the frontier."""
        return any(node.state == state for node in self.frontier)

    def empty(self):
        """Check if the frontier is empty."""
        return len(self.frontier) == 0

    def remove(self):
        """Remove and return the last node added."""
        if self.empty():
            raise Exception("Empty frontier")
        return self.frontier.pop()


class QueueFrontier(StackFrontier):
    """A queue-based frontier for breadth-first search."""
    def add(self, item):
        """Add an item to the end of the queue."""
        self.frontier.append(item)

    def remove(self):
        """Remove and return the first item added."""
        if self.empty():
            raise Exception("Empty frontier")
        return self.frontier.pop(0)

    def is_empty(self):
        """Check if the frontier is empty."""
        return len(self.frontier) == 0

    def contains_state(self, state):
        """Check if a state is in the frontier."""
        return any(node.state == state for node in self.frontier)
