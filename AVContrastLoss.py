import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['AVContrastLoss']


class AVContrastLoss(nn.Module):
    def __init__(self, temp=0.07, normalize=True):
        """
        Audio-Vision Supervised Contrastive Loss.
        Assumes visual encoder is fixed.
        
        temp : Sharpening logits by small temp.
        normalize : Normalize before compute dot product
        """
        super(AVContrastLoss, self).__init__()
        self.temp = temp
        self.normalize = normalize
    
    def forward(self, a_features, a_labels, v_features, v_labels, temp=0.07):
        
        # Asserts whether embeded to same dimension 
        assert a_features.shape[1] == v_features.shape[1]
        
        # Normalization
        if self.normalize:
            a_features = F.normalize(a_features, dim=1)
            v_features = F.normalize(v_features, dim=1)
        
        # Assumes visual encoder doesn't need gradient
        v_features = v_features.detach()
        
        # Matches audio & visual labels
        a_labels = a_labels.view(-1, 1)
        label_match = torch.eq(a_labels, v_labels.T).float()
        
        # Computes similarity for all audio & visual feature pairs
        logits = torch.einsum('ak, vk -> av', a_features, v_features)
        logits /= temp
        
        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= -logits_max.detach()
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * label_match
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 6e-7)

        # Compute mean of log-likelihood over positive (same labels)
        mean_log_prob_pos = (label_match * log_prob).sum(dim=1) / label_match.sum(dim=1)

        # loss
        loss = -mean_log_prob_pos
        loss = loss.mean()
        if math.isnan(loss.item()):
            breakpoint()

        return loss
