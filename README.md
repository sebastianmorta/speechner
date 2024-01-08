# speech-ner
 

# The script files
- **DatasetMap.py** - load and mapping the dataset 
- **Training.py** - train and save the Text model and Audio-Text model
  - _**DATASETFRAC**_ is what percentage of whole dataset will be used
  - Two outputs are generated for both models:
    - `./TorchResult/{model.model_name}{DATASETFRAC}p.pth` for _torch.save_
    - `./FromPretrainedResult/{model.model_name}{DATASETFRAC}p` for _save_pretrained_
  
- **Evaluation.py** 
  - load dataset from `./MappedDataset` 
  - load configs from `./FromPretrainedResult`
  - load models from `./TorchResult`
  - evaluate models by ```evaluator(dataset['validation'], text_tokenizer, model)```
- **AudioTextNER.py** - contains Audio-Text model and evaluation functions for this model
- **TextNER.py** - contains Text model and evaluation functions for this model
- **PreprocessDataset.py** - contains functions to map dataset
# Script usage
  - For train both models you must run ```Training.py```
  - For evaluate both models you must run ```Evaluation.py```


 
# Data source
https://drive.google.com/drive/folders/1wdilxgE5sM-Pq7bmdrqxkcyqSWY5iTp5?usp=sharing
 
# Environment
Tested on Python 3.9. Required packages listed in requirements.txt
 