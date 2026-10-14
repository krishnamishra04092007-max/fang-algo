"""
Self-Supervised Stance Loss for FANG
===================================
Models user-news interaction attitudes (support, deny, report, etc.) in a self-supervised
manner using naturally occurring social engagement signals without manual annotations.

Mathematical formulation:
1. User projection:  h_u = W_u * z_u + b_u  (shape: n_stances x n_hidden)
2. News projection:  h_n = W_n * z_n + b_n  (shape: n_stances x n_hidden)
3. Scaled matching:  logit_s = sum_k (h_u[s, k] * h_n[s, k]) / sqrt(n_hidden)
4. Loss:             L_stance = CrossEntropyLoss(logits, stance_labels)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional


class StanceClassifier(nn.Module):
    """
    Scaled bilinear projection network that predicts stance classification logits
    between user and news embeddings.

    Args:
        embed_dim: Embedding dimension of user and news nodes [default: 16]
        n_stances: Number of stance classes (e.g. 4: support_neutral, support_negative, deny, report) [default: 4]
        n_hidden: Hidden projection dimension per stance class [default: 4]
    """
    def __init__(
        self,
        embed_dim: int = 16,
        n_stances: int = 4,
        n_hidden: int = 4
    ):
        super(StanceClassifier, self).__init__()
        self.embed_dim = embed_dim
        self.n_stances = n_stances
        self.n_hidden = n_hidden

        self.user_proj = nn.Linear(embed_dim, n_stances * n_hidden)
        self.news_proj = nn.Linear(embed_dim, n_stances * n_hidden)
        self.norm_coeff = float(np.sqrt(n_hidden))

    def forward(
        self,
        user_embed: torch.Tensor,
        news_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes stance prediction logits.

        Args:
            user_embed: Engaged user representations, shape: (batch_size, embed_dim)
            news_embed: News representations (broadcast/aligned), shape: (batch_size, embed_dim)

        Returns:
            stance_logits: Predicted stance logits, shape: (batch_size, n_stances)
        """
        # Linear projections into (n_stances * n_hidden)
        h_u = self.user_proj(user_embed)
        h_n = self.news_proj(news_embed)

        # Bilinear dot product across the hidden dimension per stance
        e = (h_u * h_n).view(-1, self.n_stances, self.n_hidden)
        stance_logits = e.sum(dim=-1) / self.norm_coeff
        return stance_logits


class SelfSupervisedStanceLoss(nn.Module):
    """
    Computes cross-entropy loss for self-supervised stance prediction.
    """
    def __init__(
        self,
        classifier: Optional[StanceClassifier] = None,
        embed_dim: int = 16,
        n_stances: int = 4,
        n_hidden: int = 4
    ):
        super(SelfSupervisedStanceLoss, self).__init__()
        self.classifier = classifier if classifier is not None else StanceClassifier(embed_dim, n_stances, n_hidden)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        news_embeddings: torch.Tensor,
        user_embeddings: torch.Tensor,
        stance_labels: torch.Tensor,
        num_users_per_news: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Computes stance prediction loss.

        Args:
            news_embeddings: News embeddings. If num_users_per_news is provided,
                             shape is (N_news, embed_dim) and news embeddings will be repeated.
                             Otherwise shape must already match user_embeddings (N_interactions, embed_dim).
            user_embeddings: Engaged user embeddings, shape: (N_interactions, embed_dim)
            stance_labels: Ground truth stance indices, shape: (N_interactions,)
            num_users_per_news: Optional list of interaction counts per news article

        Returns:
            Scalar cross entropy loss tensor
        """
        if num_users_per_news is not None:
            repeats = torch.tensor(num_users_per_news, dtype=torch.long, device=news_embeddings.device)
            aligned_news_embed = torch.repeat_interleave(news_embeddings, repeats, dim=0)
        else:
            aligned_news_embed = news_embeddings

        logits = self.classifier(user_embeddings, aligned_news_embed)
        loss = self.cross_entropy(logits, stance_labels)
        return loss
