import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np

from dataset import ClassifierDataset
from test_dataset import TestDataset
from model.resnet34 import ResNet34
from model.cc_resnet import resnet34
from utils.metric import auroc_score
from utils.checkpoints import load_checkpoint

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
conditions = ['GENDER', 'HCC18', 'HCC22', 'HCC85', 'HCC96', 'HCC108', 'HCC111', 'CalciumScoring_AbdominalAgatston', 'AGE', 'RAF']
num_classes = len(conditions) - 2

def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='data', help='Directory with data')
    parser.add_argument('--checkpoint_path', default='checkpoints', help='Checkpoint path')
    parser.add_argument('--out_path', default='output', help='Prediction output path')
    parser.add_argument('--size', default=256, type=int, help='Size of CT to generate')
    parser.add_argument('--age_norm', default=100.0, type=float, help='Normalization of age')
    parser.add_argument('--raf_norm', default=10.0, type=float, help='Normalization of RAF')
    parser.add_argument('--only_pred', default=False, action='store_true', help='Only generate predictions or get score')
    parser.add_argument('--calc_stats', default=False, action='store_true', help='Calculate specificity and sensitivity')
    args = parser.parse_args()
    return args

def get_prediction(code, label, thresh=0.5):
    if label == 'GENDER':
        return 'female' if code <= thresh else 'male'
    elif 'HCC' in label:
        return 'ABSENT' if code <= thresh else 'PRESENT'

def calculate_accuracy(y_pred, y_label):
    pred_conditions = y_pred[:, :-2]
    label_conditions = y_label[:, :-2].long()
    batch, num_classes = label_conditions.shape
    stat = torch.zeros((4, num_classes))
    correct = torch.zeros(num_classes)
    for i in range(num_classes):
        start = i * 3
        end = (i + 1) * 3
        pred = torch.max(F.softmax(pred_conditions[:, start:end], dim=-1), dim=-1)[1]
        correct[i] = (pred == label_conditions[:, i]).sum()
        if correct[i]:
            if pred == 0:
                stat[3, i] += 1
            else:
                stat[2, i] += 1
        else:
            if pred == 0:
                stat[1, i] += 1
            else:
                stat[0, i] += 1
    return stat, correct

def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_bootstraps=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_fn(y_true[indices], y_pred[indices])
        scores.append(score)
    sorted_scores = np.sort(scores)
    lower = np.percentile(sorted_scores, ((1.0 - ci) / 2.0) * 100)
    upper = np.percentile(sorted_scores, (1 - (1.0 - ci) / 2.0) * 100)
    return lower, upper

def test(model, test_dataloader, only_pred, stat):
    with torch.no_grad():
        result = {'STUDY': []}
        for c in conditions:
            result[c] = []
        class_score_auroc = torch.zeros(num_classes)
        class_score_auprc = torch.zeros(num_classes)
        stats = torch.zeros((4, num_classes))
        correct = torch.zeros(num_classes)

        y_true = torch.LongTensor()
        y_pred = torch.FloatTensor()

        for i, val in tqdm(enumerate(test_dataloader)):
            if only_pred:
                img = val
            else:
                img, labels = val
                labels = labels.to(device)

            img = img.to(device)
            img = img.repeat(1, 3, 1, 1)
            result['STUDY'].append(test_dataloader.dataset.at(i))

            prediction = torch.sigmoid(model(img))

            if stat:
                s, c = calculate_accuracy(prediction, labels)
                stats += s
                correct += c

            if not only_pred:
                y_true = torch.cat((y_true, labels[:, :-2].cpu()), dim=0)
                y_pred = torch.cat((y_pred, prediction[:, :-2].cpu()), dim=0)

            result['AGE'].append(prediction[:, -2].item() * args.age_norm)
            result['RAF'].append(prediction[:, -1].item() * args.raf_norm)
            result['GENDER'].append(prediction[:, 0].item())

            for j in range(1, num_classes):
                result[conditions[j]].append(prediction[:, j].item())

        auroc_ci, auprc_ci = [], []
        for j in range(num_classes):
            class_score_auroc[j] = roc_auc_score(y_true[:, j], y_pred[:, j])
            class_score_auprc[j] = average_precision_score(y_true[:, j], y_pred[:, j])
            lower_auc, upper_auc = bootstrap_metric_ci(y_true[:, j].numpy(), y_pred[:, j].numpy(), roc_auc_score)
            lower_prc, upper_prc = bootstrap_metric_ci(y_true[:, j].numpy(), y_pred[:, j].numpy(), average_precision_score)
            auroc_ci.append((lower_auc, upper_auc))
            auprc_ci.append((lower_prc, upper_prc))

    return result, class_score_auroc, class_score_auprc, auroc_ci, auprc_ci, stats, correct / len(test_dataloader.dataset)

args = arg_parse()

print('Init model')
model = ResNet34(num_classes=num_classes + 2)
load_checkpoint(args.checkpoint_path, model)
model.to(device)
model.eval()

print('Initializing')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.55001191,), (0.18854326,))])
if args.only_pred:
    test_dataset = TestDataset(args.data_dir, transforms=transform, size=args.size)
else:
    test_dataset = ClassifierDataset(args.data_dir, conditions[1:-2], transforms=transform, size=args.size, train=False, age_norm=args.age_norm, raf_norm=args.raf_norm)

test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)

print('Testing')
results, auroc, auprc, auroc_ci, auprc_ci, stats, correct = test(model, test_dataloader, args.only_pred, args.calc_stats)

if not args.only_pred:
    print('AUROC and AUPRC Scores with 95% Confidence Interval:')
    for i in range(num_classes):
        print(f'{conditions[i]}:')
        print(f'  AUROC : {auroc[i]:.4f} [{auroc_ci[i][0]:.4f}, {auroc_ci[i][1]:.4f}]')
        print(f'  AUPRC : {auprc[i]:.4f} [{auprc_ci[i][0]:.4f}, {auprc_ci[i][1]:.4f}]')

if args.calc_stats:
    print('Stats [sensitivity, specificity]')
    for i in range(num_classes):
        sensitivity = stats[2, i] / (stats[2, i] + stats[1, i])
        specificity = stats[3, i] / (stats[3, i] + stats[0, i])
        print('{}:\t{:.4f}\t{:.4f}'.format(conditions[i], sensitivity, specificity))
    print(stats)

df = pd.DataFrame(results, columns=['STUDY'] + conditions)
df.to_csv(os.path.join(args.out_path, 'out.csv'), index=False)
