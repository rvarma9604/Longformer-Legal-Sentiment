import pickle
import argparse
import torch
import torch.optim as optim
from model import LongDocClassifier
from long_doc_data import Data
from torch.utils.data import DataLoader


def eval_model(args):
    '''data loader'''
    test_files = load_file_names(args.test_files)
    test_dataset = Data(test_files, infer=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    '''model setup'''
    model = LongDocClassifier()
    
    print(f'Initializing model from {args.init}')
    checkpoint = torch.load(args.init)
    model.load_state_dict(checkpoint['model_state_dict'])

    if torch.cuda.is_available():
        print('Using gpu')
        device = torch.device('cuda:1')
    else:
        print('Cannot evaluate without GPU')
        sys.exit()

    model.to(device) 
    predictions = model.predict(test_loader, device)
    model.to('cpu')

    return predictions

def main():
    parser = argparse.ArgumentParser(description='Lonformer trainer')
    parser.add_argument('test_files', help='list of test file paths pkl format')
    parser.add_argument('out_dir', help='directory to store snapshots')
    parser.add_argument('init', help='.pt file containing model weights')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    args = parser.parse_args()
    print(args)

    '''start training'''
    predictions = eval_model(args)
    with open(args.out_dir + '/predictions.pkl', 'rb') as f:
        pickle.dump(predictions, f)

    print('Finished!')


if __name__=='__main__':
    main()
