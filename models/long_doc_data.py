import torch
from torch.utils.data import Dataset
from transformers import LongformerTokenizer


class Data(Dataset):
    def __init__(self, files, max_len):
        self.files = files
        self.len = len(files)
        self.max_len = max_len
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

    def __getitem__(self, index):
        f_name = files[index]
        if 'pos' in f_name:
            label = torch.tensor(1).long()

        with open(f_name, 'r') as f:
            document = f.read.splitlines()
            document = " ".join(data)
        
        encoded_document = tokenizer(document, 
                                     add_special_tokens=True,
                                     max_length=self.max_length,
                                     padding='max_length',
                                     truncation=True
                            )

        input_ids = encoded_document['input_ids'][0]
        att_mask = encoded_document['attention_mask'][0]
        global_mask = torch.zeros_like(att_mask)
        global_mask[0] = 1

        return input_ids, att_mask, global_mask, label

    def __len__(self):
        return self.len
