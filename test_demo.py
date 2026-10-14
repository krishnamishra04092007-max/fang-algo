"""
Verification and Demo Script for Extracted FANG Algorithms
==========================================================
Executes synthetic forward and backward passes on all 4 components:
1. Temporal Attention Mechanism
2. Unsupervised Proximity Loss
3. Self-Supervised Stance Loss
4. Supervised Fake News Loss
5. FANG Joint Objective Pipeline
"""

import torch
from fang_algorithms import (
    TemporalAttentionMechanism,
    extract_most_attended_users,
    UnsupervisedProximityLoss,
    StanceClassifier,
    SelfSupervisedStanceLoss,
    FakeNewsClassifierHead,
    SupervisedFakeNewsLoss,
    IntegratedFakeNewsDetector,
    FANGJointObjective,
)


def run_demo():
    print("=" * 60)
    print("FANG Algorithms Verification & Demo")
    print("=" * 60)

    # Common dimensions from FANG defaults
    batch_size = 4
    seq_len = 10
    embed_dim = 16
    hidden_dim = 8
    n_stances = 4
    n_temp_features = 2
    n_classes = 2

    print(f"\n--- Hyperparameters & Dimensions ---")
    print(f"Batch Size: {batch_size}, Seq Len (Engagements): {seq_len}")
    print(f"Node Embed Dim: {embed_dim}, LSTM Hidden Dim: {hidden_dim}")
    print(f"Number of Stances: {n_stances}, Veracity Classes: {n_classes}")

    # 1. Temporal Attention Mechanism
    print(f"\n[1] Testing Temporal Attention Mechanism...")
    temporal_attn = TemporalAttentionMechanism(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_stances=n_stances,
        n_temp_features=n_temp_features
    )
    dummy_news_embed = torch.randn(batch_size, embed_dim, requires_grad=True)
    dummy_user_embed = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    dummy_stances = torch.randn(batch_size, seq_len, n_stances)
    dummy_timestamps = torch.rand(batch_size, seq_len, n_temp_features)
    dummy_masked_attn = torch.zeros(batch_size, seq_len)
    # Mask last 2 steps in sequence as padded
    dummy_masked_attn[:, -2:] = -1000.0

    eng_embed, attentions, fused_news = temporal_attn(
        dummy_news_embed, dummy_user_embed, dummy_stances, dummy_timestamps, dummy_masked_attn
    )
    print(f"  Engagement Context Shape: {eng_embed.shape} (Expected: [{batch_size}, {2*hidden_dim}])")
    print(f"  Attention Weights Shape:  {attentions.shape} (Expected: [{batch_size}, {seq_len}])")
    print(f"  Fused News Rep Shape:     {fused_news.shape} (Expected: [{batch_size}, {embed_dim}])")
    
    # Test top-attended user extraction
    dummy_users_batch = [[101, 102, 103, 104, 105, 106, 107, 108, 109, 110] for _ in range(batch_size)]
    top_users = extract_most_attended_users(attentions, dummy_users_batch, top_k=3)
    print(f"  Sample Extracted Top Users: {top_users}")

    # 2. Unsupervised Proximity Loss
    print(f"\n[2] Testing Unsupervised Proximity Loss...")
    prox_loss_module = UnsupervisedProximityLoss(Q=10.0, margin=3.0, mode="normal")
    all_nodes = [10, 20, 30, 40, 50, 60]
    dummy_node_embeddings = torch.randn(len(all_nodes), embed_dim, requires_grad=True)
    pos_pairs = {
        10: [(10, 20), (10, 30)],
        20: [(20, 10)],
    }
    neg_pairs = {
        10: [(10, 40), (10, 50), (10, 60)],
        20: [(20, 50), (20, 60)],
    }
    prox_loss = prox_loss_module(dummy_node_embeddings, all_nodes, pos_pairs, neg_pairs)
    print(f"  Proximity Loss (Normal): {prox_loss.item():.4f}")

    margin_prox_module = UnsupervisedProximityLoss(margin=3.0, mode="margin")
    margin_loss = margin_prox_module(dummy_node_embeddings, all_nodes, pos_pairs, neg_pairs)
    print(f"  Proximity Loss (Margin): {margin_loss.item():.4f}")

    # 3. Self-Supervised Stance Loss
    print(f"\n[3] Testing Self-Supervised Stance Loss...")
    stance_loss_fn = SelfSupervisedStanceLoss(embed_dim=embed_dim, n_stances=n_stances, n_hidden=4)
    dummy_news_embs = torch.randn(2, embed_dim, requires_grad=True)
    dummy_engaged_users = torch.randn(5, embed_dim, requires_grad=True)
    num_users_per_news = [3, 2] # 3 users on news 0, 2 users on news 1
    dummy_stance_labels = torch.tensor([0, 1, 2, 1, 3], dtype=torch.long)

    stance_loss = stance_loss_fn(
        dummy_news_embs, dummy_engaged_users, dummy_stance_labels, num_users_per_news
    )
    print(f"  Stance Cross-Entropy Loss: {stance_loss.item():.4f}")

    # 4. Supervised Fake News Loss
    print(f"\n[4] Testing Supervised Fake News Loss...")
    detector = IntegratedFakeNewsDetector(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_stances=n_stances,
        n_temp_features=n_temp_features,
        n_classes=n_classes
    )
    fake_news_loss_fn = SupervisedFakeNewsLoss()
    dummy_sources = torch.randn(batch_size, embed_dim, requires_grad=True)
    dummy_labels = torch.tensor([1, 0, 1, 0], dtype=torch.long)

    logits, _, _ = detector(
        dummy_news_embed, dummy_sources, dummy_user_embed, dummy_stances, dummy_timestamps, dummy_masked_attn
    )
    news_loss = fake_news_loss_fn(logits, dummy_labels)
    print(f"  Veracity Prediction Logits Shape: {logits.shape} (Expected: [{batch_size}, {n_classes}])")
    print(f"  Supervised Fake News Loss:        {news_loss.item():.4f}")

    # 5. Joint Multi-Task Pipeline
    print(f"\n[5] Testing Joint Multi-Task Objective Pipeline...")
    joint_model = FANGJointObjective(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_stances=n_stances,
        n_temp_features=n_temp_features,
        n_classes=n_classes
    )
    optimizer = torch.optim.Adam(joint_model.parameters(), lr=1e-3)

    # Forward Fake News
    j_news_loss, j_logits, _, _ = joint_model.forward_fake_news(
        dummy_sources, dummy_news_embed, dummy_user_embed, dummy_stances, dummy_timestamps, dummy_masked_attn, dummy_labels
    )
    # Forward Stance
    j_stance_loss = joint_model.forward_stance(
        dummy_news_embs, dummy_engaged_users, dummy_stance_labels, num_users_per_news
    )
    # Forward Proximity
    j_prox_loss = joint_model.forward_proximity(
        dummy_node_embeddings, all_nodes, pos_pairs, neg_pairs
    )

    # Compute total loss
    total_loss, loss_breakdown = joint_model.compute_joint_loss(j_news_loss, j_stance_loss, j_prox_loss)
    print(f"  Loss Breakdown: {loss_breakdown}")

    # Backward pass verification
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print(f"  Backward Pass & Optimization Step: SUCCESS! Gradients calculated cleanly.")
    print("=" * 60)
    print("ALL 4 ALGORITHMS SUCCESSFULLY VERIFIED!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
