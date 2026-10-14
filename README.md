# FANG Modular Algorithms

This folder isolates the core algorithmic components of the **FANG (Factual News Graph)** framework into clean, self-contained, and reusable PyTorch modules.

---

## Folder Structure

```
fang_algorithms/
├── __init__.py                 # Exports all modules and utilities
├── temporal_attention.py       # Algorithm 1: Temporal Attention Mechanism (Bi-LSTM + Aligned & Meta Attention)
├── proximity_loss.py           # Algorithm 2: Unsupervised Proximity Loss (Normal Skip-Gram & Margin-based)
├── stance_loss.py              # Algorithm 3: Self-Supervised Stance Loss (Bilinear Projection & Cross-Entropy)
├── fake_news_loss.py           # Algorithm 4: Supervised Fake News Loss (Classification Head & Cross-Entropy)
├── joint_pipeline.py           # Multi-task joint objective harness (L_total = L_proximity + L_stance + L_news)
├── test_demo.py                # Standalone test verification with synthetic tensors
└── README.md                   # Mathematical formulations, parameter details, and usage guide
```

---

## 1. Temporal Attention Mechanism (`temporal_attention.py`)

### Core Concept
Models how user interactions evolve over time to distinguish viral fake news patterns from steady news propagation. Combines:
1. **Aligned Attention**: Measures content alignment between user engagement and the news article.
2. **Meta Attention**: Measures the significance of interaction timing and stance.

### Mathematical Formulation
1. Engagement representation:
   $$x_t = [z_{\text{user}, t} \,\|\, s_t \,\|\, \tau_t] \in \mathbb{R}^{d_{\text{embed}} + n_{\text{stances}} + 2}$$
2. Contextualization via 2-layer Bidirectional LSTM:
   $$H = \text{BiLSTM}(X) \in \mathbb{R}^{B \times T \times (2 \cdot d_{\text{hidden}})}$$
3. Aligned attention:
   $$\alpha_{\text{aligned}} = \text{Softmax}\left( (H W_{\text{aligned}}^\top) z_{\text{news}}^\top + M_{\text{attn}} \right)$$
4. Metadata attention:
   $$\alpha_{\text{meta}} = \text{Softmax}\left( [s \,\|\, \tau] W_{\text{meta}}^\top + M_{\text{attn}} \right)$$
5. Dual-attention blending & pooling:
   $$\alpha = \frac{\alpha_{\text{aligned}} + \alpha_{\text{meta}}}{2}, \quad e_{\text{eng}} = \sum_{t=1}^T \alpha_t H_t$$
6. Contextualized news representation:
   $$e_{\text{news\_rep}} = z_{\text{news}} + e_{\text{eng}}$$

### Usage
```python
from fang_algorithms import TemporalAttentionMechanism

temporal_attn = TemporalAttentionMechanism(
    embed_dim=16,
    hidden_dim=8,
    n_stances=4,
    n_temp_features=2
)
eng_embed, attentions, fused_news = temporal_attn(
    news_embed, user_embed, stances, timestamps, masked_attn
)
```

---

## 2. Unsupervised Proximity Loss (`proximity_loss.py`)

### Core Concept
Preserves the topological and social structural proximity across entities (users, sources, news) in the representation space without requiring class labels.

### Mathematical Formulation
* **Normal Mode (Cosine Skip-Gram with Negative Sampling)**:
  $$\text{pos\_score}(u, v) = \log \sigma(\cos(z_u, z_v))$$
  $$\text{neg\_score}(u, v') = Q \cdot \mathbb{E}_{v' \in \mathcal{N}_u}[\log \sigma(-\cos(z_u, z_{v'}))]$$
  $$\mathcal{L}_{\text{proximity}} = \frac{1}{|\mathcal{B}|} \sum_{u \in \mathcal{B}} \left( -\mathbb{E}_{(u, v) \in \mathcal{P}_u}[\text{pos\_score}(u, v)] - \text{neg\_score}(u) \right)$$

* **Margin Mode (Hinge Ranking)**:
  $$\mathcal{L}_{\text{margin}}(u) = \max\left(0, \; \max_{v' \in \mathcal{N}_u} \log \sigma(\cos(z_u, z_{v'})) - \min_{v \in \mathcal{P}_u} \log \sigma(\cos(z_u, z_v)) + \text{MARGIN}\right)$$

### Key Parameters
- $Q = 10.0$ (Negative sampling multiplier)
- $\text{MARGIN} = 3.0$
- Neighborhood expansion walk depths: User = 2, Source/News = 1

### Usage
```python
from fang_algorithms import UnsupervisedProximityLoss

prox_loss_fn = UnsupervisedProximityLoss(Q=10.0, mode="normal")
loss = prox_loss_fn(embeddings, node_ids, pos_pairs, neg_pairs)
```

---

## 3. Self-Supervised Stance Loss (`stance_loss.py`)

### Core Concept
Learns to predict user attitude toward news articles (e.g. *Support Neutral*, *Support Negative*, *Deny*, *Report*) using naturally occurring interactions (tweets/quotes/replies) without manual annotation.

### Mathematical Formulation
1. Projection into stance-specific latent spaces:
   $$h_u = W_u z_u + b_u \in \mathbb{R}^{n_{\text{stances}} \times d_{\text{hidden}}}$$
   $$h_n = W_n z_n + b_n \in \mathbb{R}^{n_{\text{stances}} \times d_{\text{hidden}}}$$
2. Scaled bilinear dot product matching:
   $$\hat{y}_{u, n}^{(s)} = \frac{1}{\sqrt{d_{\text{hidden}}}} \sum_{k=1}^{d_{\text{hidden}}} h_{u, s, k} \cdot h_{n, s, k}$$
3. Stance cross-entropy loss:
   $$\mathcal{L}_{\text{stance}} = -\frac{1}{|\mathcal{E}_{\text{stance}}|} \sum_{(u, n)} \log \left( \frac{\exp(\hat{y}_{u, n}^{(y)})}{\sum_{s'} \exp(\hat{y}_{u, n}^{(s')})} \right)$$

### Usage
```python
from fang_algorithms import StanceClassifier, SelfSupervisedStanceLoss

stance_loss_fn = SelfSupervisedStanceLoss(embed_dim=16, n_stances=4, n_hidden=4)
loss = stance_loss_fn(news_embeddings, user_embeddings, stance_labels, num_users_per_news)
```

---

## 4. Supervised Fake News Loss (`fake_news_loss.py`)

### Core Concept
The final downstream veracity classifier combining source credibility, news article representation, and aggregated social engagement context.

### Mathematical Formulation
1. Concatenation:
   $$e_{\text{final}} = [z_{\text{source}} \,\|\, (z_{\text{news}} + e_{\text{eng}})] \in \mathbb{R}^{d_{\text{embed}} + 2 \cdot d_{\text{hidden}}}$$
2. Prediction:
   $$\hat{z}_{\text{news}} = W_{\text{out}} \cdot e_{\text{final}} + b_{\text{out}} \in \mathbb{R}^2$$
3. Loss:
   $$\mathcal{L}_{\text{news}} = \text{CrossEntropyLoss}(\hat{z}_{\text{news}}, y_{\text{veracity}})$$

### Usage
```python
from fang_algorithms import IntegratedFakeNewsDetector, SupervisedFakeNewsLoss

detector = IntegratedFakeNewsDetector(embed_dim=16, hidden_dim=8, n_classes=2)
loss_fn = SupervisedFakeNewsLoss()

logits, attentions, fused_news = detector(news_embed, source_embed, user_embed, stances, timestamps, masked_attn)
loss = loss_fn(logits, labels)
```

---

## 5. Joint Multi-Task Objective (`joint_pipeline.py`)

Coordinates the full training objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{proximity}} + \mathcal{L}_{\text{stance}} + \mathcal{L}_{\text{news}}$$

```python
from fang_algorithms import FANGJointObjective

joint_model = FANGJointObjective(
    embed_dim=16,
    hidden_dim=8,
    n_stances=4,
    n_temp_features=2,
    n_classes=2
)
optimizer = torch.optim.Adam(joint_model.parameters(), lr=1e-3)

# 1. Fake News Loss
news_loss, logits, attn, _ = joint_model.forward_fake_news(
    sources, news, users, stances, times, mask, labels
)

# 2. Stance Loss
stance_loss = joint_model.forward_stance(
    news_emb, user_emb, stance_labels, num_users_per_news
)

# 3. Proximity Loss
prox_loss = joint_model.forward_proximity(
    all_embeddings, all_node_ids, pos_pairs, neg_pairs
)

# 4. Joint Loss & Optimization
total_loss, metrics = joint_model.compute_joint_loss(news_loss, stance_loss, prox_loss)
total_loss.backward()
optimizer.step()
```

---

## Running the Verification Test
To run the self-contained verification suite on synthetic tensors:
```bash
python -m fang_algorithms.test_demo
```
