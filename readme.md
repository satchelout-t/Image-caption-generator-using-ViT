# Image Caption Generator using Vision Transformer (ViT)

An end-to-end image captioning pipeline using a pretrained ViT encoder and LSTM decoder, trained on the Flickr8k dataset. Built as Stage 2 of a two-part project — first implementing ViT from scratch, then applying it to caption generation.

---

## Architecture
- **Encoder**: `google/vit-base-patch16-224` pretrained on ImageNet-21k, fully frozen during training
- **Decoder**: 2-layer LSTM with 512 hidden units, trained from scratch
- **Inference**: Beam search with beam width 5 for cleaner caption generation
- **Vocabulary**: 4,097 tokens built from Flickr8k captions (min frequency: 3)
- **Dataset**: Flickr8k — 8,000 images, 40,455 image-caption pairs (5 captions per image)
- **Trainable Parameters**: 8.7M (decoder only)

---

## Results

### Loss Curve
![Loss Curve](loss_curve.png)

| Epoch | Avg Loss |
|-------|----------|
| 1 | 7.80 |
| 2 | 5.20 |
| 3 | 3.54 |
| 4 | 3.32 |
| 5 | 3.18 |
| 6 | 3.09 |

### Generated Captions

| Image | Generated Caption |
|-------|-----------------|
| ![](samples/dog_running.png) | "a dog running" |
| ![](samples/man_climbing.png) | "a man climbing mountain" |
| ![](samples/two_dogs.png) | "two dogs in field" |
| ![](samples/group_mountain.png) | "a group people on mountain" |

---

## Why Transfer Learning

The original ViT paper (*An Image is Worth 16x16 Words*, Dosovitskiy et al. 2020) showed that ViT needs massive pretraining data to work well — Google trained theirs on JFT-300M (300 million images). By using a pretrained ViT and only training the LSTM decoder on top, we get rich visual features immediately while keeping compute requirements manageable on a local GPU.

The frozen ViT encoder extracts a CLS token — a single 768-dimensional vector summarizing the entire image through self-attention over all 16×16 patches. This vector is projected and fed into the LSTM as the starting context for caption generation.

---

## Project Background

This is Stage 2 of a two-part learning process:

**Stage 1** → Built Vision Transformer completely from scratch using PyTorch for MNIST classification, achieving 88.9% accuracy. [vit-from-scratch repo](https://github.com/satchelout-t/vit-from-scratch)

**Stage 2** → Replaced the classification head with an LSTM decoder to generate captions instead of predicting classes. Same ViT architecture, different output.

---

## Setup

```bash
git clone https://github.com/satchelout-t/Image-caption-generator-using-ViT.git
cd Image-caption-generator-using-ViT
pip install torch torchvision transformers Pillow matplotlib
```

Download Flickr8k from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and structure as:
project/
├── flickr8k/
│   ├── Images/
│   └── captions.txt
├── caption_vit.ipynb

Then run all cells in `caption_vit.ipynb`.

---

## Future Upgrades

- **COCO dataset**: Already supported in code — set `USE_COCO = True` and point to COCO paths
- **Transformer decoder**: Replace LSTM with a Transformer decoder using cross-attention over all patch tokens instead of just CLS token — closer to how BLIP works
- **Feature caching**: Run ViT encoder once, save CLS vectors to disk, train LSTM on cached features — significantly faster training
- **BLEU evaluation**: Add proper BLEU score computation for quantitative comparison
- **Attention visualization**: Visualize which image patches the model attends to while generating each word