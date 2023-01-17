import torch
import argparse

from torchvision.datasets import cifar
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from resnet_2 import resnet110

parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--epochs', default=160, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
parser.add_argument('--log_name', type=str, default='./log/default', help='log name')
parser.add_argument('--warmup', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--case', type=int, default=0, help="Network case\n"
                                                        "0: original\n"
                                                        "1: BN after addition\n"
                                                        "2: ReLU before addition\n"
                                                        "3: ReLU-only pre-activation\n"
                                                        "4: full pre-activation")
parser.set_defaults(augment=True)

args = parser.parse_args()
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.set_device(args.gpu)

print()
print(args)
logger = SummaryWriter(log_dir=args.log_name)


def build_dataset(dataset):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                          (4, 4, 4, 4), mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    if dataset == "cifar10":
        train_data = cifar.CIFAR10('./CIFAR10', train=True, transform=train_transform, download=True)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.prefetch, pin_memory=True)
        val_data = cifar.CIFAR10('./CIFAR10', train=False, transform=val_transform, download=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.prefetch)
    return train_loader, val_loader


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    return correct.sum().item() / batch_size * 100


def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    return accuracy, test_loss


def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        y_f = model(inputs)
        Loss = F.cross_entropy(y_f, targets.long())
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        prec_train = accuracy(y_f.data, targets.long().data)
        train_loss += Loss.item()

        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'Prec@1 %.2f\t' % (
                      epoch, args.epochs, batch_idx + 1, len(train_loader.dataset) / args.batch_size,
                      (train_loss / (batch_idx + 1)),
                      prec_train))

    return prec_train, train_loss / (len(train_loader))


def adjust_learning_rate(optimizer, epochs):
    lr = args.lr * ((10 ** int(epochs >= args.warmup)) * (0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 120)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


train_loader, val_loader = build_dataset(args.dataset)
model = resnet110(args.case).to(device)
# print(model)
optimizer_model = torch.optim.SGD(model.parameters(), args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)

if __name__ == "__main__":
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer_model, epoch)
        train_acc, train_loss = train(model, train_loader, optimizer_model, epoch)
        logger.add_scalar("Loss/Train", train_loss, global_step=epoch)
        logger.add_scalar("Accuracy/Train", train_acc, global_step=epoch)
        test_acc, test_loss = test(model, val_loader)
        logger.add_scalar("Loss/Validation", test_loss, global_step=epoch)
        logger.add_scalar("Accuracy/Validation", test_acc, global_step=epoch)

        if best_acc < test_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.log_name + "/best.pth")
    # 保存模型
    torch.save(model.state_dict(), args.log_name + "/last.pth")
    print("Best Acc: {:.4f}%".format(best_acc))
    logger.add_text("Best Acc", str(best_acc), global_step=0)
