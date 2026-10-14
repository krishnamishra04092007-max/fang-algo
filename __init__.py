"""
FANG Algorithms Modular Package
===============================
Extracted algorithms from the FANG (Factual News Graph) framework:
- Temporal Attention Mechanism: Bi-LSTM with Aligned & Meta Attention
- Unsupervised Proximity Loss: Graph topology preservation via negative sampling
- Self-Supervised Stance Loss: Bilinear projection for user-news attitude prediction
- Supervised Fake News Loss: Verification head and cross-entropy loss
- Joint Objective Pipeline: Multi-task training harness
"""

from .temporal_attention import (
    TemporalAttentionMechanism,
    extract_most_attended_users,
)

from .proximity_loss import (
    UnsupervisedProximityLoss,
    ProximitySampler,
    LRUCache,
)

from .stance_loss import (
    StanceClassifier,
    SelfSupervisedStanceLoss,
)

from .fake_news_loss import (
    FakeNewsClassifierHead,
    SupervisedFakeNewsLoss,
    IntegratedFakeNewsDetector,
)

from .joint_pipeline import (
    FANGJointObjective,
)

__all__ = [
    "TemporalAttentionMechanism",
    "extract_most_attended_users",
    "UnsupervisedProximityLoss",
    "ProximitySampler",
    "LRUCache",
    "StanceClassifier",
    "SelfSupervisedStanceLoss",
    "FakeNewsClassifierHead",
    "SupervisedFakeNewsLoss",
    "IntegratedFakeNewsDetector",
    "FANGJointObjective",
]
