import os
import random
import shutil

from pathlib import Path

datasets = {'pretrain4w': [343, 10, 147], 'siamese': [10, 10, 480]} # Total of 500
tumors = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

for dataset, sizes in datasets.items():
    print(f'Dataset {dataset} loading...')

    for tumor in tumors:
        print(f'\tDataset {dataset}, Tumor {tumor} loading...')
        
        Path(f'dataset/{dataset}/train/{tumor}').mkdir(parents=True, exist_ok=True)
        Path(f'dataset/{dataset}/validation/{tumor}').mkdir(parents=True, exist_ok=True)
        Path(f'dataset/{dataset}/test/{tumor}').mkdir(parents=True, exist_ok=True)

        
        source = f'dataset-brain\\' + tumor
        dest_train = f'dataset\\{dataset}\\train\\' + tumor
        dest_val = f'dataset\\{dataset}\\validation\\' + tumor
        dest_test = f'dataset\\{dataset}\\test\\' + tumor

        no_of_files = sum(sizes)

        files = random.sample(os.listdir(source), no_of_files)

        for idx, file_name in enumerate(files[0:sizes[0]]):
            shutil.copy(os.path.join(source, file_name), os.path.join(dest_train, f'{tumor}_{idx}.jpg'))
        for idx, file_name in enumerate(files[sizes[0]:(sizes[0]+sizes[1])]):
            shutil.copy(os.path.join(source, file_name), os.path.join(dest_val, f'{tumor}_{idx}.jpg'))
        for idx, file_name in enumerate(files[(sizes[0]+sizes[1]):(sizes[0]+sizes[1]+sizes[2])]):
            shutil.copy(os.path.join(source, file_name), os.path.join(dest_test, f'{tumor}_{idx}.jpg'))


        print(f'\tDataset {dataset}, Tumor {tumor} complete!')


    print(f'Dataset {dataset} complete!')

