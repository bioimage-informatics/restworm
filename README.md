## restworm

### Data set preparation
#### Training data
restworm restores images in Input folder by using images in Ground_truth folder.
The folders are designated by the train_input_folder and train_gt_folder in the restworm.py.
Paired images must be stored in each folder with the same name.

#### Test data


### Dependencies
restworm relies on the following excellent packages:
- python=3.7.10
- tensorflow-gpu==1.14.0
- keras=2.3.1
- opencv
- tifffile
