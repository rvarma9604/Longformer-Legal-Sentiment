import torch
import torch.nn as nn
from transformers import LongformerForSequenceClassification


class LongDocClassifier(nn.Module):
    def __init__(self):
        super(LongDocClassifier, self).__init__()
        self.model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096')

    def forward(self, input_ids, attention_mask, global_attention_mask):
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask, 
                          global_attention_mask=global_attention_mask)

    def criterion(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)

    def train_step(self, train_loader, optimizer, device, scheduler=None):
        self.model.train()
        total_loss = total_acc = 0
        for batch_id, (input_ids, att_masks, global_masks, labels) in enumerate(train_loader):
            input_ids, att_masks, global_masks = input_ids.to(device), att_masks.to(device), global_masks.to(device)
            labels = labels.to(device)

            labels = labels.view(-1)

            optimizer.zero_grad()

            yhat = self.model(input_ids, att_masks, global_masks)[0]
            loss = self.criterion(yhat, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            torch.cuda.empty_cache()

            total_loss += loss.item()
            total_acc += (torch.argmax(yhat, axis=1) == labels).sum()

            if (batch_id % 500 == 0) or (batch_id == len(train_loader) - 1):
                print(f'\t\tTrain iter {batch_id + 1 }/{len(train_loader)}')

        train_acc = total_acc.item() / len(train_loader.dataset)
        train_loss = total_loss / len(train_loader.dataset)

        return train_acc, train_loss

    def test_step(self, test_loader, device):
        self.model.eval()
        total_loss = total_acc = 0
        for batch_id, (input_ids, att_masks, global_masks, labels) in enumerate(test_loader):
            with torch.no_grad():
                input_ids, att_masks, global_masks = input_ids.to(device), att_masks.to(device), global_masks.to(device)
                labels = labels.to(device)

                labels = labels.view(-1)

                yhat = self.model(input_ids, att_masks, global_masks)[0]
                loss = self.criterion(yhat, labels)

                torch.cuda.empty_cache()

                total_loss += loss.item()
                total_acc += (torch.argmax(yhat, axis=1) == labels).sum()

                if (batch_id % 500 == 0) or (batch_id == len(test_loader) - 1):
                    print(f'\t\tTest iter {batch_id + 1 }/{len(test_loader)}')

        test_acc = total_acc.item() / len(test_loader.dataset)
        test_loss = total_loss / len(test_loader.dataset)

        return test_acc, test_loss

    def predict(self, test_loader, device):
        self.model.eval()

        preds = None
        for batch_id, (input_ids, att_masks, global_mask) in enumerate(test_loader):
            with torch.no_grad():
                input_ids, att_masks, global_masks = input_ids.to(device), att_masks.to(device), global_masks.to(device)

                yhat = self.model(input_ids, att_masks, global_masks)[0]
                yhat = torch.argmax(yhat, axis=1)

                torch.cuda.empty_cache()

                if (batch_id % 500 == 0) or (batch_id == len(test_loader) - 1):
                    print(f'\t\tEval iter {batch_id + 1}/{len(test_loader)}')

                preds = torch.cat((preds, yhat), 0) if preds is not None else yhat

        return preds
