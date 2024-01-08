# speech-ner
 
2 scripts that train the following prediction models:
- linear

 
They predict 4 next values based on the previous 192 values (2 days) (you can change it by passing optional arguments).
The results are saved in "./results" directory.
 
# The script files
- **DatasetMap.py** - load and mapping the dataset 
- **Training.py** - train and save the Text model and Audio-Text model
  - DATASETFRAC is what percentage of whole dataset will be used
  - Two outputs are generated for both models:
    - `./TorchResult/{model.model_name}{DATASETFRAC}p.pth` for _torch.save_
    - `./FromPretrainedResult/{model.model_name}{DATASETFRAC}p` for _save_pretrained_
  -
- **Evaluation.py** 
  - load dataset 
  - load configs and models
  - evaluate models
 
# Script usage
  
## Positional arguments
- path to the .csv data
- (for one_feature.py) target column name, e.g. "1001.PM2.5[calibrated]"
- number of training epochs, e.g. 50
- batch_size, e.g. 64 
- early stopping patience, e.g. 3
 
## Example usages
```
python one_feature.py "./data/ml" "1001.PM2.5[calibrated]" 50 64 3
python all_features.py "./data/ml" 100 32 5
```
 
# Data source
https://drive.google.com/drive/folders/1wdilxgE5sM-Pq7bmdrqxkcyqSWY5iTp5?usp=sharing
 
# Environment
Tested on Python 3.9. Required packages listed in requirements.txt
 