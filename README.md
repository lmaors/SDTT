## SENTENCE-ADAPTIVE VIDEO THUMBNAIL DYNAMIC GENERATION BASED ON TRANSFORMER
The code is running in Cuda10.2 pytorch1.5.

#### Dataset
The dataset is shown in `cd dataset/annotated_thumbnail`

#### Environment
The packages `pytorch` `tqdm` `bert4keras` is necessary.

#### Extract features
`cd pre`
Run the script to extract video and sentence features.
The pre-trained weight file about ALBERT and C3D model will be upload soon.

#### Train
Just run `python run.py` for training.
Modify the config.py to adjustment parameters.
