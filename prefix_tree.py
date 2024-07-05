import pandas as pd
from transformers import  AutoTokenizer
from SFT.SFT_dataloader import Trie_str

class Trie:
    def __init__(self, token_ids, no_subsets=False) -> None:
        self.max_height = max([len(one) for one in token_ids])

        root = {}
        for token_ids in token_ids:
            level = root
            for tidx, token_id in enumerate(token_ids):
                if token_id not in level:
                    level[token_id] = {}

                level = level[token_id]

        if no_subsets and self.has_subsets(root, token_ids):
            raise ValueError(
                "Each list in `token_ids` can't be a complete subset of another list, but is"
                f" {token_ids}."
            )

        self.trie = root

    def has_subsets(self, trie, nested_token_ids):
        leaf_count = self.count_leaves(trie)
        return len(nested_token_ids) != leaf_count

    def next_tokens(self, current_seq):  # 这个只返回一层对应的token

        if len(current_seq) == 0:
            return list(self.trie.keys())

        start = self.trie

        for current_token in current_seq:
            start = start[current_token]

        next_tokens = list(start.keys())

        return next_tokens

    def reached_leaf(self, current_seq):
        next_tokens = self.next_tokens(current_seq)

        return len(next_tokens) == 0

    def count_leaves(self, root):
        next_nodes = list(root.values())
        if len(next_nodes) == 0:
            return 1
        else:
            return sum([self.count_leaves(nn) for nn in next_nodes])

    def print_trie(self, node=None, indent="", level=0):
        if node is None:
            node = self.trie
        for key, child in node.items():
            print(f"{indent}[{level}] {key}:")
            self.print_trie(child, indent + "  ", level + 1)

def single_child_count(trie):
    def count(node):
        if not node:
            return 0, 0
        if len(node) == 1:
            single_count = 1
        else:
            single_count = 0
        total_single_counts = sum(count(child)[1] for child in node.values())
        return single_count, single_count + total_single_counts
    return count(trie)[1]

def total_nodes(trie):
    def count(node):
        return 1 + sum(count(child) for child in node.values()) if node else 0
    return count(trie) - 1  # Subtract 1 to exclude the root node from the count

def nodes_by_level(trie):
    level_nodes = {}
    def traverse(node, level=0):
        if level not in level_nodes:
            level_nodes[level] = 0
        level_nodes[level] += 1
        for child in node.values():
            traverse(child, level + 1)
    traverse(trie)
    return level_nodes

# 统计prefix tree的一些特质

item_path_dict = {
    'steam': '/home/wangshuo/codes/scope-rec/data/dataset/steam/items.pickle',
    'movies': '/home/wangshuo/codes/scope-rec/data/dataset/movies/items.pickle',
    'toys': '/home/wangshuo/codes/scope-rec/data/dataset/toys/items.pickle',
}
dataset_name = 'toys'
item_path = item_path_dict[dataset_name]
item_list = pd.read_pickle(item_path)
tokenizer = AutoTokenizer.from_pretrained('/home/wangshuo/weights/llama2/Llama-2-7b-hf-chat/')
tokenizer.add_special_tokens({'additional_special_tokens': ['<SOI>', '<EOI>', '<SEP>']})
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id
ctrl_symbols = ['<SOI>', '<EOI>']
ctrl_symbols = list(map(tokenizer.convert_tokens_to_ids, ctrl_symbols))



item_list = list(map(lambda x: x.strip(),item_list)) # list(list)
item_ids = tokenizer.batch_encode_plus(item_list,add_special_tokens=False).data['input_ids']
input_ids_append =[]
for item in item_ids:
    new_list = []
    new_list= [ctrl_symbols[0]]+item+[ctrl_symbols[1]]
    input_ids_append.append(new_list)

item_prefix_tree = Trie_str(input_ids_append)
next_token1 = item_prefix_tree.next_tokens([32002])
print(len(next_token1))
item_prefix_tree = Trie(input_ids_append)
next_token2 = item_prefix_tree.next_tokens([32002])
print(len(next_token2))
next_token1.sort()
next_token2.sort()
for i in range(len(next_token1)):
    if next_token1[i] != next_token2[i]:
        print('error')

# node_level_dict = nodes_by_level(item_prefix_tree.trie)
# print(total_nodes(item_prefix_tree.trie))
# print(item_prefix_tree.count_leaves(item_prefix_tree.trie))
# print(single_child_count(item_prefix_tree.trie))
# for key in node_level_dict.keys():
#     print(f'{node_level_dict[key]}')
