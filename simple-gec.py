import csv
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, BertForMaskedLM, BertModel, BertPreTrainedModel

from utils import seed_everything, make_data_from_txt, GECDataset, BalancedDataLoader, Batch
import transformers
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Config:
    n_epoch = 10
    batch_size = 5
    max_data_size = 1000
    show_every = 10
    model_name = 'bert-base-uncased'

class BertErrorCorrection(BertPreTrainedModel):
    def __init__ (self, vocab_size):
        from transformers import BertConfig
        from transformers import BertModel
        from transformers.modeling_bert import BertEncoder

        config = BertConfig(num_labels=vocab_size)
        super(BertErrorCorrection, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()
        for _, param in self.bert.named_parameters():
            param.requires_grad = True
        self.config = config

    def forward (self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                 position_ids=None, head_mask=None):
        bert_output = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                             attention_mask=attention_mask, head_mask=head_mask)
        logits = self.linear(bert_output[0])

        outputs = (logits,) + bert_output[2:]  # add hidden states and attention if they are here

        loss_fct = CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)).mean()
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


#net = BertForSequenceClassification.from_pretrained(pre_trained_weights, num_labels=2, output_attentions=True)
config = Config


# main here
seed_everything(1234)
tokenizer = BertTokenizer.from_pretrained(Config.model_name)
train_data = make_data_from_txt("data/fce.train.gold.bea19.m2.tsv", config.max_data_size, tokenizer)
train_ds = GECDataset(train_data)
train_dl = BalancedDataLoader(train_ds, tokenizer.pad_token_id, config.batch_size)




model = BertErrorCorrection(tokenizer.vocab_size)


def train_model (net, dataloader, optimizer, num_epochs):
    net.to(device)
    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0
            batch_processed_num = 0
            bar = tqdm(total=len(dataloader))

            # データローダーからミニバッチを取り出す
            for i, (x, y) in enumerate(dataloader):
                batch = Batch(x.to(device), y.to(device), pad=tokenizer.pad_token_id)
                # optimizerの初期化
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # 5. BERTモデルでの予測とlossの計算、backpropの実行
                    outputs = net(batch.source, token_type_ids=None, attention_mask=batch.source_mask, labels=batch.target)
                    # loss and accuracy
                    loss, logits = outputs[:2]


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    curr_loss = loss.item()
                    epoch_loss += curr_loss
                    # epoch_result["TP"] += torch.sum((preds == labels.data) * (labels.data == 1)).cpu().numpy()
                    # epoch_result["TN"] += torch.sum((preds == labels.data) * (labels.data == 0)).cpu().numpy()
                    # epoch_result["FP"] += torch.sum((preds != labels.data) * (labels.data == 0)).cpu().numpy()
                    # epoch_result["FN"] += torch.sum((preds != labels.data) * (labels.data == 1)).cpu().numpy()

                    # epoch_recall = recall_score(y_true=labels.data.cpu(), y_pred=preds.cpu(), pos_label=0)
                    # epoch_precision = precision_score(y_true=labels.data.cpu(), y_pred=preds.cpu(), pos_label=0)
                batch_processed_num += 1
                SHOW_EVERY = config.show_every
                if batch_processed_num % SHOW_EVERY == 0 and batch_processed_num != 0:
                    _, preds = torch.topk(logits, k=1)
                    preds2 = preds.squeeze(-1)
                    yText = tokenizer.convert_ids_to_tokens(preds2[0].tolist())
                    y_text_part = " ".join(yText[:min(10, len(yText))])
                    xText = tokenizer.convert_ids_to_tokens(batch.source[0].tolist())
                    x_text_part = " ".join(xText[:min(10, len(xText))])
                    bar.set_postfix_str(f'Loss: {loss.item():.5f}, pred: {y_text_part}, org: {x_text_part}')
                    bar.update(SHOW_EVERY)

#                    print("orig:" + " ".join(xText))
#                    print("pred:" + " ".join(yText))

                    # print(epoch_result)

            # loss and corrects per epoch
            epoch_loss = epoch_loss / batch_processed_num


            print('Epoch {}/{} | {:^5} | Loss:{:.4f}'.format(epoch + 1, num_epochs, phase, epoch_loss))

    return net
optimizer = optim.Adam(model.parameters(), lr=1e-4)

net_trained = train_model(model, train_dl, optimizer, num_epochs=config.n_epoch)
