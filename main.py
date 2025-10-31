import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import dataloader
from models.EEGNet import EEGNet
from models.DeepConvNet import DeepConvNet

# -------- Dataset wrapper --------
class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index, ...], dtype=torch.float32)
        label = torch.tensor(int(self.label[index]), dtype=torch.long)
        return data, label

    def __len__(self):
        return len(self.label)

# -------- Plot function --------
def plot_curve(values, title, xlabel, ylabel, outpath):
    plt.figure()
    plt.plot(range(1, len(values) + 1), values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=600, bbox_inches='tight')
    plt.close()

# -------- Training for one epoch --------
def train_for_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

# -------- Evaluation --------
def eval(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = 100.0 * correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc

# -------- main --------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='EEGNet', choices=['EEGNet', 'DeepConvNet'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--elu_alpha', type=float, default=1.0)
    parser.add_argument('--F1', type=int, default=16)
    parser.add_argument('--D', type=int, default=2)
    parser.add_argument('--kernel_len', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--save_dir', type=str, default='./result')
    args = parser.parse_args()


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("-Using device:", device)
    print("-Parameters:")
    print(f" model={args.model}, batch_size={args.batch_size}, lr={args.lr}, num_epochs={args.num_epochs}, dropout={args.dropout}, elu_alpha={args.elu_alpha}")
    print(f" F1={args.F1}, D={args.D}, kernel_len={args.kernel_len}, weight_decay={args.weight_decay}, save_dir={save_dir}")

    # -------- Load data --------
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()

    if train_data.ndim == 3:
        train_data = train_data[:, None, :, :]
        test_data = test_data[:, None, :, :]

    Chans = train_data.shape[2]
    Samples = train_data.shape[3]
    num_classes = len(np.unique(train_label))

    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # -------- Build model --------
    if args.model == 'EEGNet':
        model = EEGNet(num_classes=num_classes, Chans=Chans, Samples=Samples,
                       F1=args.F1, D=args.D, kernel_len=args.kernel_len,
                       dropoutRate=args.dropout, elu_alpha=args.elu_alpha)
    else:
        model = DeepConvNet(num_classes=num_classes, Chans=Chans, Samples=Samples,
                            dropoutRate=args.dropout, elu_alpha=args.elu_alpha)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # -------- Training loop --------
    best_test_acc = 0.0
    train_loss_list, train_acc_list = [], []
    test_loss_list, test_acc_list = [], []

    print(f"-Start training {args.model} at {timestamp}")
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train_for_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = eval(model, test_loader, criterion, device)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        print(f"[{epoch:03d}/{args.num_epochs}] Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_path = os.path.join(save_dir, f'best_{args.model}.pt')
            torch.save(model.state_dict(), best_model_path)

    best_train_acc = max(train_acc_list)
    best_train_epoch = np.argmax(train_acc_list) + 1
    best_test_epoch = np.argmax(test_acc_list) + 1

    # -------- Save log and plot --------
    df = pd.DataFrame({
        'epoch': range(1, args.num_epochs + 1),
        'train_loss': train_loss_list,
        'train_acc': train_acc_list,
        'test_loss': test_loss_list,
        'test_acc': test_acc_list
    })
    csv_path = os.path.join(save_dir, f'train_log_{args.model}.csv')
    df.to_csv(csv_path, index=False)
    print("-Log saved to:", csv_path)

    plot_curve(train_acc_list, 'Train Accuracy', 'Epoch', 'Accuracy (%)',
               os.path.join(save_dir, f'train_acc_{args.model}.png'))
    plot_curve(test_acc_list, 'Test Accuracy', 'Epoch', 'Accuracy (%)',
               os.path.join(save_dir, f'test_acc_{args.model}.png'))
    plot_curve(train_loss_list, 'Train Loss', 'Epoch', 'Loss',
               os.path.join(save_dir, f'train_loss_{args.model}.png'))
    plot_curve(test_loss_list, 'Test Loss', 'Epoch', 'Loss',
               os.path.join(save_dir, f'test_loss_{args.model}.png'))

    print(f"-Best train accuracy: {best_train_acc:.2f}% at epoch: {best_train_epoch}")
    print(f"-Best test accuracy: {best_test_acc:.2f}% at epoch: {best_test_epoch}")
    print(f"-Best model saved to: {best_model_path}")
