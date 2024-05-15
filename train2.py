import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset import get_datasets
from models.base_model import BaseModel
from models.two_views_base_model import TwoViewsBaseModel
from losses.loss import classification_regression_loss
from utils.utils import init, arg_parse
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


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
        'mse': F.mse_loss(y_score, y_true).item()
    }

    return metric


def evaluate(model: nn.Module, dataloader: DataLoader):
    """
    Evaluate one epoch
    :return: tuple with loss (scalar) and metrics (dict with scalars) for test-set
    """

    total_loss = 0
    all_y_true = torch.LongTensor()
    all_y_pred = torch.LongTensor()
    all_y_score = torch.FloatTensor()

    model.eval()
    with torch.no_grad():
        for x, labels in tqdm(dataloader):
            y = model(x)

            loss = classification_regression_loss(model.conditions, y, labels.to(model.device))
            total_loss += loss.item()

            y_pred = (torch.sigmoid(y) > 0.5).type(torch.float)
            all_y_true = torch.cat((all_y_true, labels.to('cpu')), dim=0)
            all_y_pred = torch.cat((all_y_pred, y_pred.to('cpu')), dim=0)
            all_y_score = torch.cat((all_y_score, y.to('cpu')), dim=0)

        total_loss /= len(dataloader)

    metrics = {}

    for i, condition in enumerate(model.conditions):
        if condition.startswith('HCC'):
            metrics[condition] = compute_classification_metrics(all_y_score[:, i], all_y_pred[:, i], all_y_true[:, i])
        else:
            metrics[condition] = compute_regression_metrics(all_y_score[:, i], all_y_true[:, i])

    return total_loss, metrics


def train(model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, args):
    """
    Train loop
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    min_loss = float('inf')
    early_stop = 0

    for epoch in range(args.epochs):

        train_loss = 0

        model.train()
        for x, labels in tqdm(train_dataloader):
            optimizer.zero_grad()
            y = model(x)
            loss = classification_regression_loss(model.conditions, y, labels.to(model.device))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        val_loss, val_metrics = evaluate(model, test_dataloader)

        print(f"Epoch: {epoch+1} \t Train Loss: {train_loss:.3f} \t Val Loss: {val_loss:.3f} \t")
        print(f"Val Metrics:")
        print(json.dumps(val_metrics, indent=4))

        if val_loss < min_loss:
            min_loss = val_loss
            early_stop = 0
        else:
            early_stop += 1

        if early_stop >= args.early_stop_criteria:
            break

    gc.collect()

    return


def run(args):
    train_dataset, test_dataset = get_datasets(args)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
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
        model = TwoViewsBaseModel(train_dataset.conditions)
    else:
        model = BaseModel(train_dataset.conditions)

    train(model, train_dataloader, test_dataloader, args)


if __name__ == '__main__':
    args = arg_parse()
    init(args)
    run(args)
