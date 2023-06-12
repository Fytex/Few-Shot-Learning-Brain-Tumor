# Few-Shot-Learning-Brain-Tumor
4-way 10-Shot Learning with a Brain Tumor (MRI) Dataset using a pre-trained model (VGG16) and a Siamese Network

Both trained models are inside website's folder. However, if you want to train them again you have to build the dataset by downloading it from https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri, put all images together into separate files by classes and executed `brain_dataset.py` to distribute the files automatically into sub-folders as we wanted.


AP_Pairs -> Siamese Network using Pairs (Contrastive Loss) 
AP_Triples -> Siamese Network using Triples (Triplet Loss)

AP_Pairs didn't obtain much better results compared to the classification problem with only a VGG16. 
AP_Triples got better results than AP_Pairs and classifcation only.

To execute the website you only need to run these commands after cloning this repository:
```
cd website
pip install -r requirements.txt
python main.py
```


For the implementation of this project we based ourselves on the repository from manos-mark: https://github.com/manos-mark/metacovid-siamese-neural-network
