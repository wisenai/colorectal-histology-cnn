# Colorectal Histology Classification

Classifying colorectal cancer tissue images using CNNs, data augmentation, and transfer learning (VGG16).

## About

This project uses the [Colorectal Histology dataset](https://www.tensorflow.org/datasets/catalog/colorectal_histology) — 5,000 H&E-stained tissue images at 150x150 pixels across 8 tissue types (tumor, stroma, immune cells, debris, mucosal glands, adipose, and background).

I built three models and compared them:
1. A baseline CNN trained from scratch
2. The same CNN after training on augmented (flipped) images
3. A VGG16 model (pre-trained on ImageNet) fine-tuned on the histology data

Transfer learning ended up working way better than the other two, which makes sense since VGG16 already knows how to pick up on edges and textures from being trained on millions of images.

## How to run

This was developed on Google Colab with a T4 GPU. To reproduce:

1. Open [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU** (`Runtime → Change runtime type → T4 GPU`)
3. Upload `colorectal_histology_cnn.py`
4. Run:
```python
!pip install tensorflow-datasets
!python colorectal_histology_cnn.py
```

Or run locally if you have a GPU:
```bash
pip install -r requirements.txt
python colorectal_histology_cnn.py
```

Plots get saved to `outputs/`.

## Model architecture

**Baseline CNN:**
```
Conv2D(8) → MaxPool → Conv2D(16) → MaxPool → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Dense(32) → Softmax(8)
```

**Transfer learning:** Took VGG16 with ImageNet weights, froze everything, and replaced the last layer with a Dense(8, softmax) for our 8 classes. Then trained just that layer on images resized to 224x224.

## Dataset reference

> Kather JN, et al. "Multi-class texture analysis in colorectal cancer histology." *Scientific Reports*, 2016.

## Acknowledgments

Built during the [InspirIT AI](https://www.inspiritai.com/) summer program.
