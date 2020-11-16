# Damage-Map Estimation After Forest Fires Using Drone Images and Deep-Learning Algorithms

## Overall
The repo contains all source-code for our proposed approach in the paper titled "Damage-Map Estimation After Forest Fires Using Drone Images and Deep-Learning Algorithms"
### Forest fire information
Location: Andong, South Korea 

Date: from April 24, 2020 to April 26, 2020
### Data acquisition information
Captured location: Mount 112, Ingeum-ri, Pungcheon-myeon, Andong-si, Gyeongsangbuk-do, 15-3 Haari, Namhu-myeon, Andong-si, Gyeongsangbuk-do

Devices: Phantom 4 Pro vB2.0

### Burned area mapping results
![](figure/map_1.PNG)

![](figure/map_2.PNG)
m
## Setup environment

Install [minianaconda](https://docs.conda.io/en/latest/miniconda.html)

`conda install -r requirements.txt`

## Sample data structure 
The given sample dataset shows how the implementation is conducted.
```
sample_data/sample_location_1_data
 |
 +-- Img
 |  |
 |  +-- img (1).png
 |  +-- img (2).png
 |  +-- ...
 +-- Label
 |  |
 |  +-- label (1).png
 |  +-- label (2).png
 |  +-- ...
 +-- Orig
 |  |
 |  +-- orig.JPG

sample_data/sample_location_2_data
 |
 +-- Img
 |  |
 |  +-- img (1).png
 |  +-- img (2).png
 |  +-- ...
 +-- Label
 |  |
 |  +-- label (1).png
 |  +-- label (2).png
 |  +-- ...
 +-- Orig
 |  |
 |  +-- orig.JPG
```

## Scripts explaination
Using [train_models](train_models.ipynb) to train the dual models.

With pretrained weights from [this link](https://drive.google.com/drive/folders/1SAv41CwAtO8iWP3WWP2_t4xNfaHiuOFr?usp=sharing)
use [predict_dual_models](predict_dual_models.ipynb) to predict sample testing dataset.

After receive the predicted results, [post processing functions](post_processing_functions.mlx) can be used to merge and overlay the results.

To create ortho-photo using commercial software. The EXIF information needed to be extracted by using [extract EXIF function](extract_EXIF.ipynb)
