import torch
from datasets import load_from_disk, load_metric
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, AutoModelForAudioClassification, \
    AutoModelForTokenClassification, BertConfig, AutoConfig, BertModel

import AudioTextNER
import TextNER

from AudioTextNER import CustomAudioTextModel
from Ontonotes import Ontonotes5Features
from TextNER import CustomTextModel
# from modelsClass import DatasetHelper, process_voxpopuli_non_normalized, evaluate_model, Ontonotes5Features

num_labels = len(Ontonotes5Features.ontonotes_labels_bio)
dataset = load_from_disk('./MappedDataset/Voxpopuli5p')
text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

textConfig = AutoConfig.from_pretrained('./Configs/bertText')

modelText = CustomTextModel(textConfig, num_labels=num_labels).to('cuda' if torch.cuda.is_available() else 'mps')
modelText.load_state_dict(torch.load('./TorchResult/TextModel5p.pth'))

audio_text_config_text = AutoConfig.from_pretrained('./Configs/bert')
audio_text_config_audio = AutoConfig.from_pretrained('./Configs/wav2vec2')

modelAudioText = CustomAudioTextModel(audio_config=audio_text_config_audio, text_config=audio_text_config_text, num_labels=num_labels)
modelAudioText.load_state_dict(torch.load('./TorchResult/AudioTextModel5p.pth'))

# models = [modelText, modelAudioText]
models = [modelAudioText, modelText]
# evaluators = [TextModelUtils.evaluate_text_model, AudioTextModelUtils.evaluate_audio_text_model]
evaluators = [AudioTextNER.evaluate_audio_text_model, TextNER.evaluate_text_model]

if __name__ == "__main__":
    for model, evaluator in zip(models, evaluators):

        model.eval()
        print(dataset['validation'].shape)
        result = evaluator(dataset['validation'], text_tokenizer, model)
        print("RESULTSS", result)
