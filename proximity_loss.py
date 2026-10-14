"""
Unsupervised Proximity Loss for FANG
===================================
Preserves the topological and social structural proximity across entities
(users, news, and sources) in the representation space.

Mathematical formulation:
1. Normal (Default, Skip-Gram with Negative Sampling):
   pos_score = log(sigmoid(cos_sim(z_u, z_pos)))
   neg_score = Q * E[log(sigmoid(-cos_sim(z_u, z_neg)))]
   L_proximity(u) = E[-pos_score - neg_score]

2. Margin-based (Alternative):
   L_margin(u) = max(0, max_neg(log(sigmoid(cos_sim))) - min_pos(log(sigmoid(cos_sim))) + MARGIN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import random
from typing import Dict, List, Tuple, Set, Optional


class LRUCache:
    """Least-Recently-Used (LRU) Cache for storing expanded graph neighborhoods."""
    def __init__(self, capacity: int = 20000):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: Set[int]) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class UnsupervisedProximityLoss(nn.Module):
    """
    Computes unsupervised graph proximity loss for batch node embeddings.
    Supports both standard negative-sampling cross-entropy (normal) and margin ranking.

    Args:
        Q: Negative sampling weight coefficient [default: 10]
        margin: Distance margin for hinge ranking loss [default: 3.0]
        mode: Loss computation mode ('normal' or 'margin') [default: 'normal']
    """
    def __init__(
        self,
        Q: float = 10.0,
        margin: float = 3.0,
        mode: str = "normal"
    ):
        super(UnsupervisedProximityLoss, self).__init__()
        self.Q = Q
        self.margin = margin
        self.mode = mode

    def forward(
        self,
        embeddings: torch.Tensor,
        nodes: List[int],
        node_positive_pairs: Dict[int, List[Tuple[int, int]]],
        node_negative_pairs: Dict[int, List[Tuple[int, int]]]
    ) -> Optional[torch.Tensor]:
        """
        Calculates proximity loss for given batch node representations.

        Args:
            embeddings: Tensor of node embeddings, shape (N_unique_nodes, embed_dim)
            nodes: List of node IDs corresponding to rows of embeddings (length N_unique_nodes)
            node_positive_pairs: Mapping node -> list of (node, pos_neighbor) tuples
            node_negative_pairs: Mapping node -> list of (node, neg_node) tuples

        Returns:
            Scalar loss tensor, or None if no valid pairs exist in batch
        """
        if self.mode == "margin":
            return self.compute_margin_loss(embeddings, nodes, node_positive_pairs, node_negative_pairs)
        else:
            return self.compute_normal_loss(embeddings, nodes, node_positive_pairs, node_negative_pairs)

    def compute_normal_loss(
        self,
        embeddings: torch.Tensor,
        nodes: List[int],
        node_positive_pairs: Dict[int, List[Tuple[int, int]]],
        node_negative_pairs: Dict[int, List[Tuple[int, int]]]
    ) -> Optional[torch.Tensor]:
        """
        Normal skip-gram proximity loss with cosine similarity and negative sampling.
        """
        node2index = {n: i for i, n in enumerate(nodes)}
        node_scores = []

        for node in node_positive_pairs:
            pps = node_positive_pairs.get(node, [])
            nps = node_negative_pairs.get(node, [])
            if len(pps) == 0 or len(nps) == 0:
                continue

            # 1. Negative pairs score: Q * E[log(sigmoid(-cos_sim))]
            neg_indices = [list(x) for x in zip(*nps)]
            neg_node_idx = [node2index[x] for x in neg_indices[0]]
            neg_neighb_idx = [node2index[x] for x in neg_indices[1]]
            
            neg_u_emb = embeddings[neg_node_idx]
            neg_v_emb = embeddings[neg_neighb_idx]
            neg_sim = F.cosine_similarity(neg_u_emb, neg_v_emb)
            neg_score = self.Q * torch.mean(torch.log(torch.sigmoid(-neg_sim)), dim=0)

            # 2. Positive pairs score: E[log(sigmoid(cos_sim))]
            pos_indices = [list(x) for x in zip(*pps)]
            pos_node_idx = [node2index[x] for x in pos_indices[0]]
            pos_neighb_idx = [node2index[x] for x in pos_indices[1]]
            
            pos_u_emb = embeddings[pos_node_idx]
            pos_v_emb = embeddings[pos_neighb_idx]
            pos_sim = F.cosine_similarity(pos_u_emb, pos_v_emb)
            pos_score = torch.log(torch.sigmoid(pos_sim))

            # 3. Combined proximity loss for anchor node
            node_score = torch.mean(-pos_score - neg_score).view(1, -1)
            node_scores.append(node_score)

        if len(node_scores) > 0:
            return torch.mean(torch.cat(node_scores, dim=0))
        return None

    def compute_margin_loss(
        self,
        embeddings: torch.Tensor,
        nodes: List[int],
        node_positive_pairs: Dict[int, List[Tuple[int, int]]],
        node_negative_pairs: Dict[int, List[Tuple[int, int]]]
    ) -> Optional[torch.Tensor]:
        """
        Margin ranking proximity loss.
        """
        node2index = {n: i for i, n in enumerate(nodes)}
        node_scores = []

        for node in nodes:
            pps = node_positive_pairs.get(node, [])
            nps = node_negative_pairs.get(node, [])
            if len(pps) == 0 or len(nps) == 0:
                continue

            # Positive similarity
            pos_indices = [list(x) for x in zip(*pps)]
            pos_u_idx = [node2index[x] for x in pos_indices[0]]
            pos_v_idx = [node2index[x] for x in pos_indices[1]]
            pos_sim = F.cosine_similarity(embeddings[pos_u_idx], embeddings[pos_v_idx])
            pos_score, _ = torch.min(torch.log(torch.sigmoid(pos_sim)), dim=0)

            # Negative similarity
            neg_indices = [list(x) for x in zip(*nps)]
            neg_u_idx = [node2index[x] for x in neg_indices[0]]
            neg_v_idx = [node2index[x] for x in neg_indices[1]]
            neg_sim = F.cosine_similarity(embeddings[neg_u_idx], embeddings[neg_v_idx])
            neg_score, _ = torch.max(torch.log(torch.sigmoid(neg_sim)), dim=0)

            # Hinge margin loss: max(0, neg_score - pos_score + margin)
            score = torch.max(
                torch.tensor(0.0, device=embeddings.device),
                neg_score - pos_score + self.margin
            ).view(1, -1)
            node_scores.append(score)

        if len(node_scores) > 0:
            return torch.mean(torch.cat(node_scores, dim=0))
        return None


class ProximitySampler:
    """
    Samples positive (near) and negative (far) node pairs for proximity training.
    Uses BFS neighborhood expansion with configurable depths and an LRU cache.

    Hyperparameter defaults from FANG:
        USER_WALK_LEN = 2, SOURCE_NEWS_WALK_LEN = 1
        USER_NUM_POS = 8, USER_NUM_NEG = 8
        SOURCE_NEWS_NUM_POS = 4, SOURCE_NEWS_NUM_NEG = 4
    """
    def __init__(
        self,
        adj_lists: Dict[int, Set[str]],
        news_nodes: Set[int],
        user_nodes: Set[int],
        source_nodes: Set[int],
        user_walk_len: int = 2,
        source_news_walk_len: int = 1,
        user_num_pos: int = 8,
        user_num_neg: int = 8,
        source_news_num_pos: int = 4,
        source_news_num_neg: int = 4,
        cache_capacity: int = 20000
    ):
        self.adj_lists = adj_lists
        self.news = news_nodes
        self.users = user_nodes
        self.sources = source_nodes
        self.user_walk_len = user_walk_len
        self.source_news_walk_len = source_news_walk_len
        self.user_num_pos = user_num_pos
        self.user_num_neg = user_num_neg
        self.source_news_num_pos = source_news_num_pos
        self.source_news_num_neg = source_news_num_neg

        self.source_news_cache = LRUCache(cache_capacity)
        self.user_cache = LRUCache(cache_capacity)
        
        self.positive_pairs: List[Tuple[int, int]] = []
        self.negative_pairs: List[Tuple[int, int]] = []
        self.node_positive_pairs: Dict[int, List[Tuple[int, int]]] = {}
        self.node_negative_pairs: Dict[int, List[Tuple[int, int]]] = {}

    def reset(self):
        self.positive_pairs = []
        self.negative_pairs = []
        self.node_positive_pairs = {}
        self.node_negative_pairs = {}

    def get_neighbors(self, node: int) -> List[int]:
        neighs = self.adj_lists.get(int(node), set())
        return [int(x.split("#")[0]) for x in neighs]

    def sample_nodes(self, nodes: List[int]) -> List[int]:
        """
        Performs positive and negative pair extraction for batch nodes,
        returning the set of all unique nodes required for embedding calculation.
        """
        source_news_nodes = [n for n in nodes if n in self.sources or n in self.news]
        user_nodes = [n for n in nodes if n in self.users]

        self._get_proximity_samples(
            source_news_nodes, self.source_news_num_pos, self.source_news_num_neg,
            self.source_news_walk_len, self.source_news_cache
        )
        self._get_proximity_samples(
            user_nodes, self.user_num_pos, self.user_num_neg,
            self.user_walk_len, self.user_cache
        )

        unique_nodes = list(
            set([i for x in self.positive_pairs for i in x]) |
            set([i for x in self.negative_pairs for i in x])
        )
        return unique_nodes

    def _get_proximity_samples(
        self,
        nodes: List[int],
        num_pos: int,
        num_neg: int,
        walk_len: int,
        cache: LRUCache
    ):
        source_news = self.news | self.sources
        for node in nodes:
            homo_nodes = self.users if node in self.users else source_news
            neighbors = cache.get(node)
            if neighbors == -1:
                neighbors, frontier = {node}, {node}
                for _ in range(walk_len):
                    current = set()
                    for outer in frontier:
                        current |= set(self.get_neighbors(outer))
                    frontier = current - neighbors
                    neighbors |= current
                cache.put(node, neighbors)

            far_nodes = homo_nodes - neighbors
            actual_neighbors = neighbors - {node}

            # Sample positive pairs
            pos_samples = random.sample(actual_neighbors, num_pos) if num_pos < len(actual_neighbors) else list(actual_neighbors)
            pos_pairs = [(node, pos) for pos in pos_samples]
            self.positive_pairs.extend(pos_pairs)
            self.node_positive_pairs[node] = pos_pairs

            # Sample negative pairs
            neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else list(far_nodes)
            neg_pairs = [(node, neg) for neg in neg_samples]
            self.negative_pairs.extend(neg_pairs)
            self.node_negative_pairs[node] = neg_pairs
