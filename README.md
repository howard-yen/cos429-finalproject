# COS429 Final Project: Font Generation Using Autoencoders and Discriminator Networks

> Final project for COS429: Computer Vision. We aim to use an autoencoder and discriminator to develop a network that is able to generate a complete font when given an image of a single character. The resulting model is able to learn both the style and shape of font characters.

---

## Table of Contents

- [Installation](#installation)
- [Dataset](#Dataset)
- [Model](#Model)
- [Evaluation](#Evaluation)

---

## Installation

### Clone

- Clone this repo with

```shell
git clone https://github.com/howard-yen/cos429-finalproject
```

### Setup

- Download the required packages

```shell
pip install -r requirements.txt
```

Many of our scripts are in juypter notebooks so it's recommended to install that as well.

```shell
pip install notebook
```

or if you are using conda

```shell
conda install -c conda-forge notebook
```

You can start running juypter notebook and run our code with

```shell
juypter notebook
```

---

## Dataset

Our dataset is located in the folder /images/, where each letter is in its own folder. The correspondences between file names and font names can be found in fonts.csv.

You can also generate the images yourself by running the code in "fonttopng.ipynb", where you can also specify the image size and the font size.

---

## Model

The resulting model we trained after about 300 epochs is in the file "encdec.pt", and you can load it for more training using

```python
encdec = EncoderDecoder()
encdec.load_state_dict(torch.load(model_file))
```

You can also modify the network as well as train it from scratch by running the code in "train.ipynb".
We recommend doing the training on a GPU as it will increase the training speed significantly, and our script automatically detects if you have the proper NVIDIA drivers and cuda set up.

---

## Evaluation

You can run some evaluations on the trained model using the code in "eval.ipynb", where there are snippets of code that displays generated font images as well as calculate the losses of the model on the validation dataset.
