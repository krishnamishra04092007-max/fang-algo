"""
Temporal Attention Mechanism for FANG (Factual News Graph)
==========================================================
Captures the temporal dynamics of user engagement with news articles.
Combines content-engagement alignment (Aligned Attention) and propagation
temporal metadata (Meta Attention) to produce an aggregated social context vector.

Mathematical formulation:
1. Input per engagement t: x_t = [u_t || s_t || tau_t]
2. Contextualization: H = BiLSTM(X)
3. Aligned attention: alpha_aligned = softmax((H * W_aligned^T) * e_news^T + mask)
4. Meta attention:    alpha_meta = softmax([s_t || tau_t] * W_meta^T + mask)
5. Blended attention: alpha = (alpha_aligned + alpha_meta) / 2
6. Aggregated context: e_eng = sum_t (alpha_t * H_t)
7. Fused news rep:    e_news_rep = e_news + e_eng
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Set, Tuple, Optional


class TemporalAttentionMechanism(nn.Module):
    """
    Temporal Attention Mechanism combining Bi-LSTM sequence encoding
    with dual-attention (content alignment and metadata attention).
    
    Args:
        embed_dim: Dimension of node embeddings (news and user) [default: 16]
        hidden_dim: LSTM hidden units per direction [default: 8 -> bidirectional: 16]
        n_stances: Number of stance categories [default: 4]
        n_temp_features: Number of temporal features (mean timestamp, std timestamp) [default: 2]
        num_layers: Number of LSTM layers [default: 2]
        dropout: Dropout probability [default: 0.0]
    """
    def __init__(
        self,
        embed_dim: int = 16,
        hidden_dim: int = 8,
        n_stances: int = 4,
        n_temp_features: int = 2,
        num_layers: int = 2,
        dropout: float = 0.0
    ):
        super(TemporalAttentionMechanism, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_stances = n_stances
        self.n_temp_features = n_temp_features
        self.num_layers = num_layers
        
        # Bi-LSTM input size = user_embed_dim + stance_vector_dim + timestamp_features_dim
        lstm_input_dim = embed_dim + n_stances + n_temp_features
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        
        # Aligned Attention: projects 2*hidden_dim LSTM states to news embed_dim space
        self.aligned_attn_proj = nn.Linear(
            in_features=hidden_dim * 2,
            out_features=embed_dim,
            bias=False
        )
        
        # Meta Attention: projects stance + time features (dim = n_stances + n_temp_features) to a scalar score
        self.meta_attn_proj = nn.Linear(
            in_features=n_stances + n_temp_features,
            out_features=1,
            bias=False
        )

    def forward(
        self,
        news_embed: torch.Tensor,
        user_embed: torch.Tensor,
        stances: torch.Tensor,
        timestamps: torch.Tensor,
        masked_attn: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of Temporal Attention Mechanism.

        Args:
            news_embed: News node representation from GraphSAGE, shape: (batch_size, embed_dim)
            user_embed: Engaged user representations, shape: (batch_size, seq_len, embed_dim)
            stances: One-hot/count stance vectors, shape: (batch_size, seq_len, n_stances)
            timestamps: Scaled mean & std times, shape: (batch_size, seq_len, n_temp_features)
            masked_attn: Attention mask (0 for valid, -1000 for pad), shape: (batch_size, seq_len)

        Returns:
            eng_embed: Aggregated engagement context vector, shape: (batch_size, 2 * hidden_dim)
            attentions: Attention weights over sequence, shape: (batch_size, seq_len)
            news_embed_rep: Fused news representation (news_embed + eng_embed), shape: (batch_size, embed_dim)
        """
        # 1. Concatenate user engagement representations with metadata
        # Shape: (batch_size, seq_len, embed_dim + n_stances + n_temp_features)
        engage_inputs = torch.cat([user_embed, stances, timestamps], dim=-1)
        
        # 2. Sequence modeling via Bi-LSTM
        # hidden_states: (batch_size, seq_len, 2 * hidden_dim)
        hidden_states, _ = self.lstm(engage_inputs)
        hidden_states = self.dropout(hidden_states)
        
        # 3. Aligned attention: measures content alignment between engagement and news
        # projected_hidden_states: (batch_size, seq_len, embed_dim)
        projected_hidden_states = self.aligned_attn_proj(hidden_states)
        # aligned_attentions: (batch_size, seq_len)
        aligned_attentions = torch.bmm(
            projected_hidden_states,
            news_embed.unsqueeze(-1)
        ).squeeze(-1)
        
        # 4. Meta attention: measures significance of interaction stance and timing
        meta_inputs = torch.cat([stances, timestamps], dim=-1)
        # meta_attentions: (batch_size, seq_len)
        meta_attentions = self.meta_attn_proj(meta_inputs).squeeze(-1)
        
        # 5. Masking and Softmax normalization
        aligned_attentions = torch.softmax(aligned_attentions + masked_attn, dim=-1)
        meta_attentions = torch.softmax(meta_attentions + masked_attn, dim=-1)
        
        # 6. Combined blended attention
        # attentions: (batch_size, seq_len)
        attentions = (aligned_attentions + meta_attentions) / 2.0
        
        # 7. Attention-weighted pooling of engagement states
        # eng_embed: (batch_size, 2 * hidden_dim)
        eng_embed = torch.bmm(attentions.unsqueeze(1), hidden_states).squeeze(1)
        
        # 8. Additive fusion with news representation (requires 2*hidden_dim == embed_dim)
        news_embed_rep = torch.add(news_embed, eng_embed)
        
        return eng_embed, attentions, news_embed_rep


def extract_most_attended_users(
    attentions: Optional[torch.Tensor],
    engage_users_batch: List[List[int]],
    top_k: int = 3
) -> Set[int]:
    """
    Identifies the top-K most attended users per news article from the
    temporal attention distribution. In FANG, these users are prioritized
    during graph proximity and stance updates to keep training scalable.

    Args:
        attentions: Tensor of attention weights, shape (batch_size, seq_len)
        engage_users_batch: List of user ID lists engaged with each news article
        top_k: Number of most attended users to sample per news item [default: 3]

    Returns:
        Set of unique user IDs identified as top-attended
    """
    top_attn_users = set()
    if attentions is not None:
        attn_np = attentions.detach().cpu().numpy()
        # Sort attention indices ascending, pick last top_k (highest attention)
        attn_sorted_args = np.argsort(attn_np, axis=-1)[:, -top_k:]
        for i, news_engaged_users in enumerate(engage_users_batch):
            attn_args = attn_sorted_args[i]
            for idx in attn_args:
                if idx < len(news_engaged_users):
                    top_attn_users.add(news_engaged_users[idx])
    else:
        # Fallback: pick users with most engagements across batch
        all_users = [u for u_list in engage_users_batch for u in u_list]
        from collections import Counter
        top_attn_users = set([u for u, _ in Counter(all_users).most_common(top_k * len(engage_users_batch))])
        
    return top_attn_users
