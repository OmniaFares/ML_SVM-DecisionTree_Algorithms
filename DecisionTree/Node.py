class Node:
    def __init__(self, feature, left, right, info_gain):
        self.feature = feature
        self.left = left
        self.right = right
        self.info_gain = info_gain