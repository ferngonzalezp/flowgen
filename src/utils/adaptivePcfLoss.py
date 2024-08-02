import torch
import numpy as np


class MovingAverage:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = []

    def update(self, new_value):
        self.values.append(new_value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return self.average()

    def average(self):
        return sum(self.values) / len(self.values)

class StabilityTracker:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.scores = []

    def update(self, new_score):
        self.scores.append(new_score)
        if len(self.scores) > self.window_size:
            self.scores.pop(0)
        return self.get_stability()

    def get_stability(self):
        if len(self.scores) < 2:
            return 1.0
        return 1.0 / (1.0 + torch.var(torch.tensor(self.scores)))

class AdaptivePCFLLoss(torch.nn.Module):
    def __init__(self, num_classes=3, gamma=2.0, stability_factor=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.stability_factor = stability_factor
        self.val_averages = [MovingAverage() for _ in range(num_classes)]
        self.stability_tracker = StabilityTracker()
        self.total_epochs = None
        self.weights = [1.0] * num_classes

    def update_metrics(self, val_loss):
        smoothed_scores = [ma.update(score) for ma, score in zip(self.val_averages, val_loss)]
        avg_loss = sum(smoothed_scores) / len(smoothed_scores)
        stability = self.stability_tracker.update(avg_loss)
        return smoothed_scores, stability

    def cosine_progress(self, l):
        return torch.clamp(0.5 * (1 + torch.cos((l - 1) * np.pi)), min = 0, max=1)

    def forward(self, y_pred, y_true, labels, loss_func):
            
            # Get smoothed F1 scores and stability (these should be updated externally)
            val_scores = torch.tensor([ma.average() for ma in self.val_averages], device=y_pred.device)
            stability = torch.tensor(self.stability_tracker.get_stability(), device=y_pred.device)
            # Base weights

            w = []
            d = 0
            for i, val_score in enumerate(val_scores):
                w.append(self.cosine_progress((val_score-d)))
                d = val_score

            # Adjust weights based on val scores
            f1_avg = torch.mean(val_scores)
            w[0] *= torch.max(torch.tensor(1.0, device=y_pred.device), f1_avg / (val_scores[0] + 1e-8))
            w[1] *= torch.max(torch.tensor(1.0, device=y_pred.device), f1_avg / (val_scores[1] + 1e-8))
            w[2] *= torch.max(torch.tensor(1.0, device=y_pred.device), f1_avg / (val_scores[2] + 1e-8))

            # Apply stability adjustment
            stability_adjustment = torch.clamp(1 + (stability - 1) * self.stability_factor, min=0.1, max=10)
            w[0] = 1 + (w[0] - 1) * stability_adjustment + 1e-8
            w[1] = 1 + (w[1] - 1) * stability_adjustment + 1e-8
            w[2] = 1 + (w[2] - 1) * stability_adjustment + 1e-8

            # Normalize weights
            total_weight = torch.sum(torch.tensor(w, device=y_pred.device))
            w[0] /= total_weight
            w[1] /= total_weight
            w[2] /= total_weight

            min_weight = 0.05
            w = [max(weight, min_weight) for weight in w]
            w = [weight / sum(w) for weight in w]  # Renormalize

            # Combine weights based on class
            self.weights = w
            labels = torch.nn.functional.one_hot(labels, self.num_classes)
            weights = 0.0
            for i in range(self.num_classes):
                weights +=( w[i] * labels[:,i] + 1e-8)

            # Sample difficulty (Focal Loss component)
                with torch.no_grad():
                    base_loss = loss_func(y_pred, y_true, None, False)
                    #assert not torch.isnan(base_loss).any(), "NaN in base loss"
                    #print(f"Base loss range: {base_loss.min().item()} to {base_loss.max().item()}")
                    p_t = torch.clamp(base_loss, min=1e-8, max=1e2)
                    modulating_factor = torch.log1p(p_t) ** self.gamma

            # Apply both class weights and sample difficulty
            #weighted_loss = weights * modulating_factor * loss_func(y_pred, y_true)
            weighted_loss = loss_func(y_pred, y_true, weights * modulating_factor, True)

            #weighted_loss = torch.where(torch.isfinite(weighted_loss), weighted_loss, torch.tensor(1e6, device=y_pred.device))

            #print(f"Weights: {w}")
            #print(f"Val scores: {val_scores}")
            #print(f"Stability: {stability}")
            #print(f"Modulating factor: {modulating_factor}")
            #print(f"Weighted loss: {weighted_loss}")

            
            return weighted_loss