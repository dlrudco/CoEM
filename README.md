# CoEM
Official Code Repository for CoEM

Our CoEM is tested on pytorch 1.7.0(py3.7_cuda10.2.89_cudnn7.6.5_0)

In order to reproduce our work, you need to follow the steps below.

# Training Image G-D
Under the image_classifier folder, you can find train_GAN.ipynb file containing training routines for image VAE module.

# Training Sound G-D
Under the sound_classifier folder, you can find train_GAN.py file. 
Use the following command to train the generator-discriminator models

`python train_GAN.py --musicnn_dis`

Once you finish training, you can find pre-trained models under '**_experiments_**' folder

# Extracting features
We provide the google_drive links for extracted features we used under each folders.

You can also extract your own features via following scripts.

For Image:
  Run `AVContrast.ipynb` under image_classifier folder

For Sound:
  Run script under sound_classifier folder

  `python vector_extract.py`
  
# Training CoEM Mapper

After successful vector extraction, you may now train the CoEM mapper by executing following script

`python vector_mapping.py --learn`

# Testing CoEM Mapper

You can also test the trained mapper for two metrics

Classification Accuracy:
`python vector_mapping.py --test --resume WEIGHT_PATH`

Retrieval Accuracy:
`python retrieval_test.py --resume WEIGHT_PATH`