from time import time

import numpy as np
from datasets.dataset import ClassifierDataset, get_dataloaders
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models.resnet34 import ResNet34
from utils.checkpoints import *
from utils.loss import *
from utils.utils import arg_parse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
conditions = ['gender', 'HCC18', 'HCC22', 'HCC85', 'HCC96', 'HCC108', 'HCC111', 'age MSE', 'raf MSE']
num_classes = len(conditions) - 2


def calculate_accuracy(y_pred, y_label, threshold=0.5):
    pred_conditions = y_pred[:, :-2]
    label_conditions = y_label[:, :-2]
    pred_age = y_pred[:, -2]
    label_age = y_label[:, -2]
    pred_raf = y_pred[:, -1]
    label_raf = y_label[:, -1]
    loss = nn.MSELoss()
    batch, _ = label_conditions.shape

    correct = torch.zeros(num_classes + 2)
    one = torch.ones(1).to(device)
    zero = torch.zeros(1).to(device)
    for i in range(num_classes):
        pred_labels = torch.where(pred_conditions[:, i] > threshold, one, zero)
        correct[i] = (pred_labels == label_conditions[:, i]).sum()

    correct[-2] = loss(pred_age, label_age)
    correct[-1] = loss(pred_raf, label_raf)

    return correct


def print_accuracy(train_accuracy, test_accuracy, train_auroc, test_auroc):
    print('\t\tTrain accuracy:\tTest accuracy:\tTrain AUROC:\tTest AUROC:')
    for i, (train_acc, test_acc, train_au, test_au) in enumerate(
            zip(train_accuracy, test_accuracy, train_auroc, test_auroc)):
        print(
            '{}\t:\t{:.4f}%\t{:.4f}%\t{:.4f}\t\t{:.4f}'.format(conditions[i], train_acc * 100, test_acc * 100, train_au,
                                                               test_au))
    print('{}\t:\t{:.4f}\t\t{:.4f}'.format(conditions[-2], train_accuracy[-2], test_accuracy[-2]))
    print('{}\t:\t{:.4f}\t\t{:.4f}'.format(conditions[-1], train_accuracy[-1], test_accuracy[-1]))


def print_auroc(train_auroc, test_auroc):
    for i in range(num_classes):
        print('{}\t: {:.4f}\t\t{:.4f}'.format(conditions[i], train_auroc[i], test_auroc[i]))


def train_epoch(model, dataloader, optimizer):
    tot_loss = 0
    accuracy = torch.zeros(num_classes + 2, dtype=torch.float32)
    tot_score = 0
    class_score = torch.zeros(num_classes)
    y_score = torch.Tensor()
    y_true = torch.LongTensor()
    for i, (img, labels) in tqdm(enumerate(dataloader)):
        img = img.to(device)
        labels = labels.to(device)
        img = img.repeat(1, 3, 1, 1)

        prediction = torch.sigmoid(model(img))
        isnan = torch.isnan(prediction[:, :-2]).any()
        isinf = torch.isinf(prediction[:, :-2]).any()
        if isnan or isinf:
            print(f"NaN detected in predictions: {isnan}, Inf detected: {isinf}")

        y_score = torch.cat((y_score, prediction[:, :-2].detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, labels[:, :-2].detach().to('cpu')), dim=0)
        # s, cs = auroc_score(prediction[:, :-2], labels[:, :-2])
        # tot_score += s
        # class_score += cs

        loss = multilabel_regression_loss(prediction, labels)
        tot_loss += loss.item()

        accuracy += calculate_accuracy(prediction, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del img, labels, prediction
    accuracy /= len(dataloader.dataset)
    roc_auc = roc_auc_score(y_true, y_score, average=None)

    return tot_loss / (i + 1), accuracy, roc_auc, roc_auc  # 100 * tot_score/(i + 1), 100 * class_score/(i + 1)


def train(model, train_dataloader, test_dataloader, optimizer, scheduler, epochs):
    best_loss = np.inf

    for epoch in range(epochs):
        model.train()

        start_train_time = time()
        train_loss, train_accuracy, train_auroc, train_class_auroc = train_epoch(model, train_dataloader, optimizer)
        scheduler.step(train_loss)
        train_time = time() - start_train_time

        start_test_time = time()
        test_loss, test_accuracy, test_auroc, test_class_auroc = test(model, test_dataloader)
        test_time = time() - start_test_time

        # print('Epoch: [{}/{}], Train loss: {:.4f}, Test loss: {:.4f}, Train score: {:.2f}, Test score: {:.2f} Train time: {:.2f}s, Test time: {:.2f}s'.format(epoch+1, epochs, train_loss, test_loss, train_auroc, test_auroc, train_time, test_time))
        # print('\t\tTrain AUROC:\tTest AUROC:')
        # print_auroc(train_class_auroc, test_class_auroc)
        # print('\t\t  Train accuracy:\tTest accuracy:')
        print(f"Epoch: {epoch + 1}")
        print_accuracy(train_accuracy, test_accuracy, train_class_auroc, test_class_auroc)

        train_state = {'epoch': epoch + 1,
                       'state_dict': model.state_dict(),
                       'optim_dict': optimizer.state_dict()}
        model_state = {'state_dict': model.state_dict()}
        is_best = test_loss < best_loss
        if is_best:
            best_loss = test_loss
        save_checkpoint(train_state, model_state, is_best, args.checkpoint_dir)


def test(model, test_dataloader):
    tot_loss = 0
    accuracy = torch.zeros(num_classes + 2, dtype=torch.float32)
    tot_score = 0
    class_score = torch.zeros(num_classes)
    y_score = torch.Tensor()
    y_true = torch.LongTensor()

    model.eval()
    with torch.no_grad():
        for i, (img, labels) in enumerate(test_dataloader):
            img = img.to(device)
            labels = labels.to(device)
            img = img.repeat(1, 3, 1, 1)

            prediction = torch.sigmoid(model(img))
            y_score = torch.cat((y_score, prediction[:, :-2].detach().to('cpu')), dim=0)
            y_true = torch.cat((y_true, labels[:, :-2].detach().to('cpu')), dim=0)
            # s, cs = auroc_score(prediction[:, :-2], labels[:, :-2])
            # tot_score += s
            # class_score += cs

            loss = multilabel_regression_loss(prediction, labels)
            tot_loss += loss.item()

            accuracy += calculate_accuracy(prediction, labels)

            del img, labels, prediction

    accuracy /= len(test_dataloader.dataset)
    roc_auc = roc_auc_score(y_true, y_score, average=None)
    return tot_loss / (i + 1), accuracy, roc_auc, roc_auc  # 100 * tot_score/(i + 1), 100 * class_score/(i + 1)


args = arg_parse()

train_dataloader, test_dataloader = get_dataloaders(args)

print('Initializing')
if args.pretrain:
    print('resnet chexpert')
    model = ResNet34(2 * 14, pretrained=True).to(device)
    # print('coord conv chexpert')
    # models = resnet34(pretrained=False, num_classes=2*14).to(device)

    load_checkpoint(args.pretrain, model)
    model.resnet34.fc = nn.Linear(model.resnet34.fc.in_features, num_classes + 2)
    # head = list(models.head.layers.children())
    # in_features = 512
    # head = head[:-1]
    # models.head.layers = nn.Sequential(*head)
    # models.head.layers = nn.Sequential(models.head.layers, nn.Linear(in_features, num_classes+2))
else:
    # print('pretrained torch resnet34')
    # models = ResNet34(3*num_classes+2, pretrained=True)
    print('ResNet34 implementation')
    model = ResNet34(2 * 14, pretrained=False).to(device)
    model.resnet34.fc = nn.Linear(model.resnet34.fc.in_features, num_classes + 2)
    # models = resnet34(pretrained=False, num_classes=num_classes+2).to(device)

if (torch.cuda.device_count() > 1):
    device_ids = list(range(torch.cuda.device_count()))
    print("GPU devices being used: ", device_ids)
    model = nn.DataParallel(model, device_ids=device_ids)

model = model.to(device)

# optimizer = optim.SGD(models.parameters(), lr=args.lr, momentum=0.9)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer, args.decay_start_epoch, gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

print('Training')
train(model, train_dataloader, test_dataloader, optimizer, scheduler, args.epochs)
