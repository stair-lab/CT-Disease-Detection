import torch
import torch.nn.functional as F


def multilabel_regression_loss(y_pred, y_label):
    pred_conditions = y_pred[:, :-2]
    pred_scores = y_pred[:, -2:]
    label_conditions = y_label[:, :-2]
    label_scores = y_label[:, -2:]

    loss = F.binary_cross_entropy(pred_conditions, label_conditions)
    loss += F.mse_loss(pred_scores, label_scores)
    return loss


def classification_regression_loss(conditions, y_pred: torch.Tensor, y_label: torch.Tensor) -> torch.Tensor:
    """
    Calculate full loss for multitask (binary classifications + regressions)
    :param y_pred: tensor with predictions (after sigmoid for classification)
    :param y_label: tensor with ground truth labels
    :return: tensor with total loss value
    """
    class_ids = [i for i in range(len(conditions)) if conditions[i].startswith('HCC')]
    reg_ids = [i for i in range(len(conditions)) if not conditions[i].startswith('HCC')]
    pred_conditions = y_pred[:, class_ids]
    pred_scores = y_pred[:, reg_ids]
    label_conditions = y_label[:, class_ids]
    label_scores = y_label[:, reg_ids]

    class_loss = F.binary_cross_entropy_with_logits(pred_conditions + 1e-5, label_conditions)
    reg_loss = F.mse_loss(pred_scores, label_scores)

    total_loss = class_loss.mean() + reg_loss.mean()

    return total_loss
