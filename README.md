# X-Ray Analysis
This code was done for my Advanced Topic Presentation in EECS 649, Introduction to Artificial Intelligence. 

## Data
It was trained to analyze chest X-Rays from the [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/) to recognize the presence of Pleural Effusion. The dataset holds around 225,000 chest X-Rays from 65,000 patients, labeled with 14 different conditions.

The dataset contains frontal and side images, though the model was only trained on frontal images. These make up the majority of images anyway, so the decision probably resulted in a relatively-small decrease in accuracy.

Pleural Effusion was chosen simply because it had the most known training examples. Many (over half for some conditions) of the training samples are given an "unknown" classification, with Pleural Effusion having the fewest at around 100,000. All unknown examples were removed for training.

## Model
The model was mostly taken from the [CheXpert paper](https://arxiv.org/abs/1901.07031). That is, the DenseNet121 architecture was used, with the top layer being replaced with a dense layer. It was initialized with weights trained using ImageNet, and all weights were tuned to fit the dataset.

A sigmoid activation was applied following the top layer to convert the output to a probability, and Adam was used as an optimizer.

## Evaluation
The model was trained for 1400 iterations with batch size 16, achieving a cross-entropy loss of 0.134 and accuracy of 0.942 on the training set. The performance on the evaluation set was worse, with a loss of 0.815 and accuracy of 0.779. This likely would have improved with more training though, as the model was trained for under 1 epoch.