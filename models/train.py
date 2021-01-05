import sys
import pickle
import argparse
import torch
import torch.optim as optim
from model import LongDocClassifier
from long_doc_data import Data
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


def load_file_names(files):
    with open(train_file, 'rb') as f:
        files = pickle.load(f)

    return files

def save_snapshot(epoch, model, optimizer, scheduler, train_acc, train_loss, test_acc, test_loss, out_dir):
    torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_acc': train_acc, 'train_loss': train_loss,
                'test_acc': test_acc, 'test_loss': test_loss},
                out_dir + '/snapshot-' + str(epoch+1) + '.pt')

def train_model(model, train_loader, test_loader, optimizer, device, start_epoch, max_epochs,
                train_acc, train_loss, test_acc, test_loss, out_dir, scheduler=None):
    model.to(device)

    for epoch in range(start_epoch, max_epochs):
        print(f'Epoch {epoch+1}/{max_epochs}')
        # train
        train_acc_epoch, train_loss_epoch = model.train_step(train_loader, optimizer, device, scheduler)
        # test
        test_acc_epoch, test_loss_epoch = model.test_step(test_loader, device)

        train_acc.append(train_acc_epoch), train_loss.append(train_loss_epoch)
        test_acc.append(test_acc_epoch), test_loss.append(test_loss_epoch)

        print(f'\tTrain_acc: {train_acc_epoch}\tTrain_loss: {train_loss_epoch}')
        print(f'\tTest_acc: {test_acc_epoch}\tTest_loss: {test_loss_epoch}')

        save_snapshot(epoch, model, optimizer, scheduler, train_acc, train_loss, test_acc, test_loss, out_dir)

    model.to('cpu')

def start_train(args):
    '''data loaders'''
    train_files = load_file_names(args.train_files)
    train_dataset = Data(train_files)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_files = load_file_names(args.test_files)
    test_dataset = Data(test_files)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    '''model setup'''
    model = LongDocClassifier()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    num_train_steps = int(len(train_dataset) / args.batch_size * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    '''device allocation'''
    if torch.cuda.is_available():
        print('Using gpu')
        device = torch.device('cuda:1')
    else:
        print('Cannot train without GPU')
        sys.exit()

    train_acc, train_loss = [], []
    test_acc, test_loss = [], []
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_acc, train_loss = checkpoint['train_acc'], checkpoint['train_loss']
        test_acc, test_loss = checkpoint['test_acc'], checkpoint['test_loss']
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    '''train the model'''
    train_model(model, train_loader, test_loader, optimizer, device, start_epoch, args.epochs, 
                train_acc, train_loss, test_acc, test_loss, args.out_dir, scheduler)

def main():
    parser = argparse.ArgumentParser(description='Lonformer trainer')
    parser.add_argument('train_files', help='list of train file paths pkl format')
    parser.add_argument('test_files', help='list of test file paths pkl format')
    parser.add_argument('out_dir', help='directory to store snapshots')
    parser.add_argument('--epochs', default=100, type=int, help='maximum epochs to be performed')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--resume', default=None, help='snapshot file')
    args = parser.parse_args()
    print(args)

    '''start training'''
    start_train(args)

    print('Finished!')


if __name__=='__main__':
    main()
