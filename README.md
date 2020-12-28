# CoEM
Official Code Repository for CoEM

Our CoEM is tested on pytorch 1.7.0(py3.7_cuda10.2.89_cudnn7.6.5_0)

In order to reproduce our work, you need to follow the steps below.

<div>
<img src="https://github.com/dlrudco/CoEM/blob/master/paper_images/fig1-overview.png?raw=true" width="90%"></img>
</div>

# Training Image G-D
Under the image_classifier folder, you can find train_GAN.ipynb file containing training routines for image VAE module.

<div>
<img src="https://github.com/dlrudco/CoEM/blob/master/paper_images/i2i.png?raw=true" width="90%"></img>
</div>

# Training Sound G-D
Under the sound_classifier folder, you can find train_GAN.py file. 
Use the following command to train the generator-discriminator models

`python train_GAN.py --musicnn_dis`

Once you finish training, you can find pre-trained models under '**_experiments_**' folder

<div>
<img src="https://github.com/dlrudco/CoEM/blob/master/paper_images/s2s.png?raw=true" width="90%"></img>
</div>

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

# References
[1] Pons, Jordi, and Xavier Serra. "musicnn: Pre-trained convolutional neural networks for music audio tagging." arXiv preprint arXiv:1909.06654 (2019).

[2] Mehdi Mirza and Simon Osindero. Conditional gener-ative adversarial nets.arXiv preprint arXiv:1411.1784,2014

[3]  Ian  Goodfellow,  Jean  Pouget-Abadie,  Mehdi  Mirza,Bing  Xu,  David  Warde-Farley,  Sherjil  Ozair,  AaronCourville, and Yoshua Bengio. Generative adversarialnets. InAdvances in neural information processing sys-tems, pages 2672–2680, 2014

# Code Inspiration
Musicnn : https://github.com/ilaria-manco/music-audio-tagging-pytorch

VAE : https://github.com/julianstastny/VAE-ResNet18-PyTorch

#Contributors

<a href="https://github.com/dlrudco/CoEM/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=dlrudco/CoEM" />
</a>