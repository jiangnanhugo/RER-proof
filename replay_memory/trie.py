# https://github.com/Askomaro/python_trie

class Node(object):
    def __init__(self, is_leave=False, freq=0):
        self.children = {}                     # list of children
        self.label = {}
        self.is_leave = is_leave
        self.freq = freq

    def __repr__(self):
        return "(ch={},l={},f={})".format(self.children, self.label, self.freq)


class TrieBuffer(object):
    def __init__(self):
        self.__name__ = "Tire tree for Replay Memory"
        self.root = Node(is_leave=False, freq=0)

    def insert(self, sub_trajectory):
        """Inserts a word into a trie. sub_trajectory: list of State-action-reward tuple
        """
        current_node = self.root
        i = 0

        while i < len(sub_trajectory) and sub_trajectory[i] in current_node.label:
            j = 0
            index = sub_trajectory[i]
            label = current_node.label[index]

            while j < len(label) and i < len(sub_trajectory) and label[j] == sub_trajectory[i]:
                i += 1
                j += 1

            if j == len(label):
                current_node = current_node.children[index]
            else:
                if i == len(sub_trajectory):
                    ex_child = current_node.children[index]
                    new_child = Node(is_leave=True, freq=0)
                    remaining_label = label[j:]

                    current_node.label[index] = label[:j]
                    current_node.children[index] = new_child
                    new_child.children[remaining_label[0]] = ex_child
                    new_child.label[remaining_label[0]] = remaining_label
                else:
                    remaining_label = label[j:]
                    remaining_word = sub_trajectory[i:]

                    new_child = Node()
                    ex_child = current_node.children[index]

                    current_node.label[index] = label[:j]
                    current_node.children[index] = new_child

                    new_child.children[remaining_label[0]] = ex_child
                    new_child.label[remaining_label[0]] = remaining_label
                    new_child.label[remaining_word[0]] = remaining_word
                    new_child.children[remaining_word[0]] = Node(is_leave=True, freq=0)

                return

        if i < len(sub_trajectory):
            current_node.label[sub_trajectory[i]] = sub_trajectory[i:]
            current_node.freq += 1
            current_node.children[sub_trajectory[i]] = Node(is_leave=True, freq=0)

        else:
            current_node.is_leave = True

    def insert_trajectory(self, trajectory, L=2, stride=2):
        """ Inserts a word into a trie. trajectory: list of State-action-reward tuple
        """
        L = L * stride
        if L >= len(trajectory):
            L = len(trajectory)
        for idx in range(0, len(trajectory) - L, stride):
            # print(trajectory[idx: idx + L])
            exisited, path = self.find(trajectory[idx: idx + L])
            if not exisited:
                self.insert(trajectory[idx: idx + L])
            else:
                self.update_freq(path)
            # print(self.root)

    def update_freq(self, path):
        for node in path:
            node.freq += 1

    def find(self, sub_trajectory):
        """Returns if the sub-trajectory is in the trie.
        sub_trajectory: list of State-action-reward tuple
        """
        i = 0
        current_node = self.root
        visited_path = []
        while i < len(sub_trajectory) and sub_trajectory[i] in current_node.label:
            j = 0
            index = sub_trajectory[i]
            label = current_node.label[index]

            while j < len(label) and i < len(sub_trajectory):
                if sub_trajectory[i] != label[j]:
                    return False, None
                i += 1
                j += 1

            if j == len(label) and len(sub_trajectory) >= i:
                current_node = current_node.children[index]
            else:
                return False, None

        return i == len(sub_trajectory) and current_node.is_leave, visited_path

    def sample(self, batch_size, reversed):
        list_of_sub_trajectories = []
        for bi in range(batch_size):
            if reversed:
                list_of_sub_trajectories.append(bi)
            else:
                list_of_sub_trajectories.append(bi[::-1])
        return list_of_sub_trajectories

