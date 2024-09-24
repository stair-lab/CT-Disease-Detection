import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import warnings

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset import get_datasets
from models.base_model import BaseModel
from models.two_views_base_model import TwoViewsBaseModel
from losses.loss import classification_regression_loss
from utils.utils import init, arg_parse
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, r2_score, explained_variance_score
from torch.utils.data import random_split, Subset
from sklearn.model_selection import train_test_split
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def compute_classification_metrics(y_score, y_pred, y_true):
    metric = {
        'roc_auc': roc_auc_score(y_true, y_score),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred).item(),
        'recall': recall_score(y_true, y_pred).item(),
        'f1': f1_score(y_true, y_pred).item()
    }

    return metric


def compute_regression_metrics(y_score, y_true):
    metric = {
        'mse': F.mse_loss(y_score, y_true).item(),
        # Additional Metrics for Continuous Targets
        'r_squared': r2_score(y_true = y_true, y_pred = y_score),
        'explained_variance': explained_variance_score(y_true = y_true, y_pred = y_score)
    }

    return metric


def evaluate(model: nn.Module, dataloader: DataLoader, evaluate_metrics=True):
    """
    Evaluate one epoch
    :return: tuple with loss (scalar) and metrics (dict with scalars) for test-set
    """

    total_loss = 0
    all_y_true = torch.LongTensor()
    all_y = torch.FloatTensor()

    model.eval()
    with torch.no_grad():
        for x, labels in dataloader:
            y = model(x)

            loss = classification_regression_loss(model.conditions, y, labels.to(model.device))
            total_loss += loss.item()

            all_y_true = torch.cat((all_y_true, labels.to('cpu')), dim=0)
            all_y = torch.cat((all_y, y.to('cpu')), dim=0)

        total_loss /= len(dataloader)

    metrics = {}
    if evaluate_metrics:
        for i, condition in enumerate(model.conditions):
            if (condition.startswith('HCC') or condition.startswith('GENDER')):
                metrics[condition] = compute_classification_metrics(torch.sigmoid(all_y[:, i]), (torch.sigmoid(all_y[:, i]) > 0.5).type(torch.float), all_y_true[:, i])
            else:
                metrics[condition] = compute_regression_metrics(y_score = all_y[:, i], y_true = all_y_true[:, i])

    return total_loss, metrics


def train(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, test_dataloader: DataLoader, args):
    """
    Train loop
    """
    ckpt_name = "two_view.ckpt" if args.two_view else "one_view.ckpt"

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    min_loss = float('inf')
    early_stop = 0

    for epoch in range(args.epochs):

        train_loss = 0

        model.train()
        for x, labels in train_dataloader:
            optimizer.zero_grad()
            y = model(x)
            loss = classification_regression_loss(model.conditions, y, labels.to(model.device))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        val_loss, val_metrics = evaluate(model, val_dataloader, evaluate_metrics=True)

        print(f"Epoch: {epoch+1} \t Train Loss: {train_loss:.3f} \t Val Loss: {val_loss:.3f} \t ")

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, ckpt_name))
            early_stop = 0
        else:
            early_stop += 1

        if early_stop >= args.early_stop_criteria:
            break

    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, ckpt_name)))
    test_loss, test_metrics = evaluate(model, test_dataloader)
    print(f"Test Metrics:")
    print(json.dumps(test_metrics, indent=4))

    gc.collect()

    return


def run(args):

    train_dataset, test_dataset = get_datasets(args)

    train_idx, val_idx = train_test_split(range(len(train_dataset)), test_size=0.1, random_state=args.seed)
    train_dataset, val_dataset = Subset(train_dataset, train_idx), Subset(train_dataset, val_idx)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.val_batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    if args.two_view:
        model = TwoViewsBaseModel(train_dataset.dataset.conditions)
    else:
        model = BaseModel(train_dataset.dataset.conditions, only_last_layer=False)

    train(model, train_dataloader, val_dataloader, test_dataloader, args)


if __name__ == '__main__':
    args = arg_parse()
    init(args)
    run(args)
