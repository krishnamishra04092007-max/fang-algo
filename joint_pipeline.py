"""
Joint Training Pipeline for FANG Algorithms
==========================================
Demonstrates the end-to-end integration of all four algorithms:
1. Temporal Attention Mechanism
2. Unsupervised Proximity Loss
3. Self-Supervised Stance Loss
4. Supervised Fake News Loss

Total multi-task loss:
    L_total = L_proximity + L_stance + L_news
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from .temporal_attention import TemporalAttentionMechanism, extract_most_attended_users
from .proximity_loss import UnsupervisedProximityLoss
from .stance_loss import StanceClassifier, SelfSupervisedStanceLoss
from .fake_news_loss import FakeNewsClassifierHead, SupervisedFakeNewsLoss


class FANGJointObjective(nn.Module):
    """
    Unified multi-task objective container for FANG. Manages the joint computation
    of supervised fake news loss, self-supervised stance loss, and unsupervised proximity loss.

    Args:
        embed_dim: Dimension of node representations [default: 16]
        hidden_dim: Hidden dimension of LSTM and stance projections [default: 8]
        n_stances: Number of stance labels [default: 4]
        n_temp_features: Number of temporal features [default: 2]
        n_classes: Number of veracity classes [default: 2]
        Q: Negative sampling multiplier for proximity loss [default: 10.0]
        margin: Hinge margin for proximity loss [default: 3.0]
        proximity_mode: 'normal' or 'margin' [default: 'normal']
    """
    def __init__(
        self,
        embed_dim: int = 16,
        hidden_dim: int = 8,
        n_stances: int = 4,
        n_temp_features: int = 2,
        n_classes: int = 2,
        Q: float = 10.0,
        margin: float = 3.0,
        proximity_mode: str = "normal",
        dropout: float = 0.0
    ):
        super(FANGJointObjective, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_stances = n_stances

        # 1. Temporal Attention Mechanism
        self.temporal_attention = TemporalAttentionMechanism(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_stances=n_stances,
            n_temp_features=n_temp_features,
            num_layers=2,
            dropout=dropout
        )

        # 2. Supervised Fake News Classifier & Loss
        self.fake_news_head = FakeNewsClassifierHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_classes=n_classes
        )
        self.fake_news_loss_fn = SupervisedFakeNewsLoss()

        # 3. Self-Supervised Stance Classifier & Loss
        self.stance_classifier = StanceClassifier(
            embed_dim=embed_dim,
            n_stances=n_stances,
            n_hidden=int(hidden_dim / 2) if hidden_dim >= 4 else 4
        )
        self.stance_loss_fn = SelfSupervisedStanceLoss(classifier=self.stance_classifier)

        # 4. Unsupervised Proximity Loss
        self.proximity_loss_fn = UnsupervisedProximityLoss(
            Q=Q,
            margin=margin,
            mode=proximity_mode
        )

    def forward_fake_news(
        self,
        source_embed: torch.Tensor,
        news_embed: torch.Tensor,
        user_embed: torch.Tensor,
        stances: torch.Tensor,
        timestamps: torch.Tensor,
        masked_attn: torch.Tensor,
        news_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs Temporal Attention, produces veracity predictions, and computes fake news loss.
        """
        eng_embed, attentions, fused_news_rep = self.temporal_attention(
            news_embed=news_embed,
            user_embed=user_embed,
            stances=stances,
            timestamps=timestamps,
            masked_attn=masked_attn
        )
        logits = self.fake_news_head(source_embed, fused_news_rep)
        news_loss = self.fake_news_loss_fn(logits, news_labels)
        return news_loss, logits, attentions, fused_news_rep

    def forward_stance(
        self,
        news_embeddings: torch.Tensor,
        user_embeddings: torch.Tensor,
        stance_labels: torch.Tensor,
        num_users_per_news: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Computes self-supervised stance loss.
        """
        return self.stance_loss_fn(
            news_embeddings=news_embeddings,
            user_embeddings=user_embeddings,
            stance_labels=stance_labels,
            num_users_per_news=num_users_per_news
        )

    def forward_proximity(
        self,
        node_embeddings: torch.Tensor,
        node_ids: List[int],
        pos_pairs: Dict[int, List[Tuple[int, int]]],
        neg_pairs: Dict[int, List[Tuple[int, int]]]
    ) -> Optional[torch.Tensor]:
        """
        Computes unsupervised proximity loss.
        """
        return self.proximity_loss_fn(
            embeddings=node_embeddings,
            nodes=node_ids,
            node_positive_pairs=pos_pairs,
            node_negative_pairs=neg_pairs
        )

    def compute_joint_loss(
        self,
        news_loss: Optional[torch.Tensor],
        stance_loss: Optional[torch.Tensor],
        proximity_loss: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Sums active loss components: L_total = L_proximity + L_stance + L_news
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        loss_dict = {}

        if proximity_loss is not None:
            total_loss = total_loss + proximity_loss
            loss_dict["proximity_loss"] = float(proximity_loss.detach().cpu())
        if stance_loss is not None:
            total_loss = total_loss + stance_loss
            loss_dict["stance_loss"] = float(stance_loss.detach().cpu())
        if news_loss is not None:
            total_loss = total_loss + news_loss
            loss_dict["news_loss"] = float(news_loss.detach().cpu())

        loss_dict["total_loss"] = float(total_loss.detach().cpu())
        return total_loss, loss_dict
