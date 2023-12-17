# Vibrotactile-Modeling

This project includes scripts for data preparation, model training, testing, and folders for generated results.
！(https://github.com/C-128391/Vibrotactile-Modeling/blob/main/Vibrotactile.png)

## Data Preparation Phase

### Downsampling Data

Start by running `prepair.py` to downsample the original dataset.
```bash
python prepair.py
```

### Data Augmentation and Save
After dividing the dataset into a training set and test set, use `pre_deal.py` to augment the downsampled data and save it as an Excel file.
```bash
python pre_deal.py
```

## Training Phase

### Model Training
Use `train.py` to train the model. Adjust parameters such as epoch and batch_size based on the training environment.

```bash
python train.py
```
After training, the model will be saved in the specified directory.

## Testing Phase
Use `test.py` to validate results on the test set.
```bash
python test.py
```

## Knowledge Distillation Model
### Training the Student Model
The "generate results_teacher model" folder contains the original results generated by the model.
Use `stu-model.py` to train the student model, obtaining a complete Loss curve and the trained model.

```bash
python stu-model.py
```
View the Loss curve during training in the "t-SNE results" folder.

## Real Texture Modeling
Before using the model for real texture modeling, run `image-adjust.py` to adjust the format of images corresponding to real textures, ensuring consistency with the HaTT database's image format.

```bash
python image-adjust.py
```
Then, use the trained model to model textures.

## Result Comparison
Compare t-SNE images obtained using the student model and the teacher model. View the original images in the "t-SNE results" folder.
