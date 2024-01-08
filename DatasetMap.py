from datasets import load_from_disk, DatasetDict, load_dataset, Audio
from PreprocessDataset import preprocess_text_voxpopuli, preprocess_audio_text_voxpopuli_non_normalized

size = ['5', '10', '1%', '5%', '10%', '50%', '100%']


def DatasetHelper(from_disc, percentage=5):
    if percentage == '100%':
        dataset = load_dataset("asapp/slue", "voxpopuli")
    else:
        dataset = DatasetDict()
        dataset['train'] = load_dataset("asapp/slue", "voxpopuli", split=f"train[:{percentage}]")
        dataset['test'] = load_dataset("asapp/slue", "voxpopuli", split=f"test[:{percentage}]")
        dataset['validation'] = load_dataset("asapp/slue", "voxpopuli", split=f"validation[:{percentage}]")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


for s in size:
    dataset = DatasetHelper(False, s).map(
        lambda x: preprocess_audio_text_voxpopuli_non_normalized(x))
    print('--------TERAZ LECI--------')
    print(s)
    dataset.save_to_disk(f'./MappedDataset/Voxpopuli{s}p')
