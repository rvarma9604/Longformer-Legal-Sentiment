import os
import pickle
import argparse
import subprocess as sp
import matplotlib.pyplot as plt
import seaborn as sns
from long_doc_data import Data
from sklearn.model_selection import train_test_split
from transformers import LongformerTokenizer


def save_files(files, name):
    with open(name, 'wb') as f:
        pickle.dump(files, f)

def load_files(name):
    with open(name, 'rb') as f:
        files = pickle.load(f)
    return files

def plot(files, plt_name):
    token_lens = []
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    for f_name in files:
        with open(f_name, 'r') as f:
            document = f.read().splitlines()
        document = " ".join(document)
        
        encoded_document = tokenizer(document, add_special_tokens=True)
        token_lens.append(len(encoded_document['input_ids']))
    sns.distplot(token_lens)
    plt.savefig(plt_name, dpi=480)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Data Preparation', add_help=True)
    parser.add_argument('in_dir', help='directory containing positive and negative document directories')
    parser.add_argument('out_dir', help='directory to store the plot and train/test splits')
    parser.add_argument('--stage', default=0, type=int, help='resume processing')
    args = parser.parse_args()
    print(args)

    '''read file names'''
    if args.stage <= 0:
        files = sp.getoutput(f'find {args.in_dir} -type f').split('\n')
        pos_neg_labels = [1 if 'pos' in f_name else 0 for f_name in files]

        '''train/test split'''
        train_files, test_files = train_test_split(files, test_size=0.2, stratify=pos_neg_labels, shuffle=True)
        os.makedirs(args.out_dir, exist_ok=True)

        save_files(train_files, args.out_dir + '/train.pkl')
        save_files(test_files, args.out_dir + '/test.pkl')

        print('Split Done')

    '''tokenize and plot data'''
    if args.stage <= 1:
        train_files = load_files(args.out_dir + '/train.pkl')
        test_files = load_files(args.out_dir + '/test.pkl')

        plot(train_files, args.out_dir + '/train_token_dist.png')
        plot(test_files, args.out_dir + '/test_token_dist.png')
        
        print('Distribution Plotted')

    print('Finished!')

if __name__ == '__main__':
    main()
