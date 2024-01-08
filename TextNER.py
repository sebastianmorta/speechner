from tqdm import tqdm
from transformers import BertPreTrainedModel, BertModel, AutoConfig, PreTrainedModel, AutoModel, \
    AutoModelForTokenClassification
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from Ontonotes import Ontonotes5Features
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

class CustomTextModel(BertPreTrainedModel):
    config_class = AutoConfig

    def __init__(
            self,
            config,
            num_labels
    ):
        super(CustomTextModel, self).__init__(config=config)
        self.model_name = 'TextModel'
        self.bert = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_labels = num_labels
        self.text_dim = config.hidden_size

    def forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None
    ):
        text_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output

        logits = self.classifier(text_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, self.num_labels)
            if len(labels) % logits.shape[0] == 0:
                labels = labels.view(logits.shape[0], -1)
                labels = labels[:, 0]
                loss = loss_fct(logits, labels)
            else:
                raise ValueError("Liczba etykiet nie jest podzielna przez rozmiar wsadu")

        return SequenceClassifierOutput(loss=loss, logits=logits)


def evaluate_text_model(dataset, tokenizer, model):
    true_y = []
    pred_y = []
    for item in tqdm(dataset):
        tokenized_item = tokenize_adjust_labels(item, tokenizer)
        gold = tokenized_item['labels']
        del tokenized_item['labels']

        with torch.no_grad():
            output = model(**tokenized_item)
            logits = output.logits.to(device)
            predictions = torch.argmax(logits, dim=0)

        true_y.append(gold[1:-1])
        pred_y.append(predictions.tolist()[1:-1])
    return [true_y, pred_y]


def tokenize_adjust_labels(example, tokenizer):
    tokenized_samples = tokenizer.encode_plus(example['tokens'], is_split_into_words=True, return_tensors="pt").to(device)
    tokenized_samples.pop("token_type_ids", None)
    prev_wid = -1
    word_ids_list = tokenized_samples.word_ids()
    existing_label_ids = example["tags"]
    i = -1
    adjusted_label_ids = []

    for wid in word_ids_list:
        if (wid is None):
            adjusted_label_ids.append(-100)
        elif (wid != prev_wid):
            i = i + 1
            adjusted_label_ids.append(existing_label_ids[i])
            prev_wid = wid
        else:
            adjusted_label_ids.append(existing_label_ids[i])

    tokenized_samples["labels"] = adjusted_label_ids
    return tokenized_samples
