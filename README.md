# Vibrotactile-Modeling

This project includes scripts for data preparation, model training, testing, and folders for generated results.

## Data Preparation Phase

### Downsampling Data

Start by running `prepair.py` to downsample the original dataset.
```bash
python prepair.py
```

### Data Augmentation and Save
After dividing the dataset into a training set and test set, use pre_deal.py to augment the downsampled data and save it as an Excel file.
```bash
python pre_deal.py
```
