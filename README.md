## restworm
<p>
  <img height="200px" src="/Picture/example.png">
</p>

The restworm restores low-quality images that were acquired by sub-optimal imaging settings.
It is optimized for images of the nematode worm, <I>C. elegans</I>.
The used deep convolutional network is described in the network.py file.
The training and test methods including the preparation of data are described below. 

### Training 
The restworm restores images in Input folder by using corresponding images in Ground_truth folder.
The folders are designated by the train_input_folder and train_gt_folder in the restworm.py.
Paired images must be stored in each folder with the same name.

### Test 
A trained network is applied for images in Test folder.
The filenames are arbitrary. The predicted results are saved in the test_pred_folder.

### Dependencies
The restworm relies on the following excellent packages:
- python=3.7.10
- tensorflow-gpu==1.14.0
- keras=2.3.1
- opencv
- tifffile
