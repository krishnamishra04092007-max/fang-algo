"""
Supervised Fake News Loss & Classification Head for FANG
========================================================
Downstream supervised fake news prediction module. Combines source credibility
(source embedding), content representation (news embedding), and temporal
social engagement context (aggregated via temporal attention) to classify
news veracity (Fake vs. Real).

Mathematical formulation:
1. Joint representation: e_final = [z_source || (z_news + e_eng)]
2. Veracity logits:      z_logits = W_out * e_final + b_out
3. Veracity loss:        L_news = CrossEntropyLoss(z_logits, news_labels)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .temporal_attention import TemporalAttentionMechanism


class FakeNewsClassifierHead(nn.Module):
    """
    Linear classification head operating on concatenated source and contextualized news vectors.

    Args:
        embed_dim: Embedding dimension of source and news [default: 16]
        hidden_dim: LSTM hidden units per direction used in temporal attention [default: 8]
        n_classes: Number of target classes [default: 2: Fake=0, Real=1]
    """
    def __init__(
        self,
        embed_dim: int = 16,
        hidden_dim: int = 8,
        n_classes: int = 2
    ):
        super(FakeNewsClassifierHead, self).__init__()
        # Input size: source_embed (embed_dim) + news_rep (embed_dim = 2 * hidden_dim)
        self.input_dim = embed_dim + (2 * hidden_dim)
        self.output_layer = nn.Linear(self.input_dim, n_classes)

    def forward(
        self,
        source_embed: torch.Tensor,
        fused_news_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes fake news classification logits.

        Args:
            source_embed: Source/publisher node embeddings, shape: (batch_size, embed_dim)
            fused_news_embed: News representation fused with temporal social context, shape: (batch_size, 2 * hidden_dim)

        Returns:
            logits: Classification logits, shape: (batch_size, n_classes)
        """
        combined = torch.cat([source_embed, fused_news_embed], dim=-1)
        logits = self.output_layer(combined)
        return logits


class SupervisedFakeNewsLoss(nn.Module):
    """
    Supervised Cross-Entropy Loss for fake news classification.
    """
    def __init__(self):
        super(SupervisedFakeNewsLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted class logits, shape: (batch_size, n_classes)
            labels: Ground truth binary labels, shape: (batch_size,)

        Returns:
            Scalar cross-entropy loss tensor
        """
        return self.criterion(logits, labels)


class IntegratedFakeNewsDetector(nn.Module):
    """
    Complete end-to-end model combining Temporal Attention Mechanism with
    the Fake News Classifier Head.

    Args:
        embed_dim: Dimension of node embeddings [default: 16]
        hidden_dim: Bi-LSTM hidden units per direction [default: 8]
        n_stances: Number of stance features [default: 4]
        n_temp_features: Number of temporal features [default: 2]
        n_classes: Number of target veracity classes [default: 2]
        dropout: Dropout rate [default: 0.0]
    """
    def __init__(
        self,
        embed_dim: int = 16,
        hidden_dim: int = 8,
        n_stances: int = 4,
        n_temp_features: int = 2,
        n_classes: int = 2,
        dropout: float = 0.0
    ):
        super(IntegratedFakeNewsDetector, self).__init__()
        self.temporal_attention = TemporalAttentionMechanism(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_stances=n_stances,
            n_temp_features=n_temp_features,
            num_layers=2,
            dropout=dropout
        )
        self.classifier_head = FakeNewsClassifierHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_classes=n_classes
        )

    def forward(
        self,
        news_embed: torch.Tensor,
        source_embed: torch.Tensor,
        user_embed: torch.Tensor,
        stances: torch.Tensor,
        timestamps: torch.Tensor,
        masked_attn: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs full forward pass from node embeddings & engagement sequences to veracity logits.

        Returns:
            logits: News veracity logits, shape: (batch_size, n_classes)
            attentions: Temporal attention weights, shape: (batch_size, seq_len)
            fused_news_rep: Contextualized news representation, shape: (batch_size, embed_dim)
        """
        eng_embed, attentions, fused_news_rep = self.temporal_attention(
            news_embed=news_embed,
            user_embed=user_embed,
            stances=stances,
            timestamps=timestamps,
            masked_attn=masked_attn
        )
        logits = self.classifier_head(source_embed, fused_news_rep)
        return logits, attentions, fused_news_rep
