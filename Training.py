import argparse

from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, AutoProcessor, BertModel, TrainingArguments, Trainer, \
    DataCollatorWithPadding, BertPreTrainedModel, AutoConfig, AutoTokenizer, AutoFeatureExtractor, \
    AutoProcessor, AutoModel, Wav2Vec2Processor, Wav2Vec2ForCTC, PreTrainedModel, PreTrainedTokenizer, Wav2Vec2Model, \
    AutoTokenizer, TrainingArguments, Trainer, Wav2Vec2Tokenizer, BertTokenizer, BertConfig, \
    DataCollatorForTokenClassification
import os
import AudioTextNER
import TextNER
from PreprocessDataset import DatasetHelper, preprocess_text_voxpopuli, preprocess_audio_text_voxpopuli_non_normalized
from Ontonotes import Ontonotes5Features

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import torch.nn as nn

DATASETFRAC = '5'
path_to_dataset = f'./MappedDataset/Voxpopuli{DATASETFRAC}p'

torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_labels = len(Ontonotes5Features.ontonotes_labels_bio)
audio_config = AutoConfig.from_pretrained("facebook/wav2vec2-base-960h")
text_config = BertConfig.from_pretrained("bert-base-uncased")

text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data_collator = DataCollatorForTokenClassification(tokenizer=text_tokenizer)

modelText = TextNER.CustomTextModel(text_config, num_labels)
modelAudioText = AudioTextNER.CustomAudioTextModel(audio_config, text_config, num_labels)

models = [modelAudioText, modelText]
# models = [modelText, modelAudioText]
evaluators = [AudioTextNER.evaluate_audio_text_model, TextNER.evaluate_text_model]
# evaluators = [TextNER.evaluate_text_model, AudioTextNER.evaluate_audio_text_model]

dataset = load_from_disk(path_to_dataset)

if __name__ == "__main__":

    for model, eval in zip(models, evaluators):
        training_args = TrainingArguments(
            output_dir=f"./results/TrainResult{model.model_name}{DATASETFRAC}p",
            evaluation_strategy="steps",
            num_train_epochs=8,
            per_device_train_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            save_strategy='no',
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=text_tokenizer,
            data_collator=data_collator,
        )
        trainer.train()
        # result = eval(dataset['validation'], text_tokenizer, model)
        model.save_configs(f'./Configs/')
        torch.save(model.state_dict(), f'./TorchResult/{model.model_name}{DATASETFRAC}p.pth')
        model.save_pretrained(f'./FromPretrainedResult/{model.model_name}{DATASETFRAC}p')
