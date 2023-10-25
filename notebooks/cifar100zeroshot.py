
# %%
import numpy as np
import torch
from tqdm.notebook import tqdm
from pkg_resources import packaging

print("Torch version:", torch.__version__)


# %%
# # Loading the model
import clip

clip.available_models()

# %%
model, preprocess = clip.load("ViT-B/16")
#model.cuda().eval()
###???
# %%
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)


# %% [markdown]
#loading dataset
import os
from datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

device = "cuda" if torch.cuda.is_available() else "cpu"

cifar100 = CIFAR100(os.path.expanduser("/home/monica/Documents/CLIP/data/"), transform=preprocess, download=True, train=True)
cifar100_test = CIFAR100(os.path.expanduser("/home/monica/Documents/CLIP/data/"), transform=preprocess, download=True, train=False)

templates = [f"This is a photo of a {label}" for label in cifar100.classes] #old imagenet_templates
#text = clip.tokenize(candidate_captions).to(device) #
classnames =cifar100.classes
batch_size = 128
# batch_size = 32
data_loader = torch.utils.data.DataLoader(cifar100,
                                          batch_size=batch_size,
                                          shuffle=False)

imagenet = ImageNet("/home/monica/Documents/CLIP/data/", split='val', transform=preprocess)


# %%
templates80 = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

templates7 = '''itap of a {}.
a bad photo of the {}.
a origami {}.
a photo of the large {}.
a {} in a video game.
art of the {}.
a photo of the small {}.'''.split('\n')

templates = [
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a photo of {}.' 
]

#%%
import tqdm

@torch.no_grad()
def zeroshot_classifier(classnames, templates):
    zeroshot_weights = []
    for classname in tqdm.tqdm(classnames): #qui potrei usare classi a caso per fallo totalmente unsup
      texts = [template.format(classname) for template in templates] #format with class
      texts = clip.tokenize(texts).cuda() #tokenize
      class_embeddings = model.encode_text(texts) #embed with text encoder
      class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
      class_embedding = class_embeddings.mean(dim=0)
      class_embedding /= class_embedding.norm()
      zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

zeroshot_weights = zeroshot_classifier(cifar100.classes, templates7)
#zeroshot_weights_ = zeroshot_classifier(cifar100.classes, templates80)
#zeroshot_weightss_ = zeroshot_classifier(cifar100.classes, templates)
zeroshot_weights.shape
#, zeroshot_weights_.shape, zeroshot_weightss_.shape

# %%
# # Zero-shot prediction
def accuracy(output, target, topk=(1,)):
  pred = output.topk(max(topk), 1, True, True)[1].t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  return pred[0], [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

all_data, all_labels,all_true_labels = None, None, None
with torch.no_grad():
  top1, top5, n = 0., 0., 0.
  #, top1_, top5_, top1s, top5s, n = 0., 0., 0., 0., 0., 0., 0.
  for i, (images, target) in enumerate(tqdm.tqdm(data_loader)):
    images = images.cuda()
    target = target.cuda()
    #all_data = images if all_data is None else torch.cat([all_data, images])
    #all_true_labels = target if all_true_labels is None else torch.cat([all_true_labels, target])
    # predict
    image_features = model.encode_image(images)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    # measure accuracy
    logits = 100. * image_features @ zeroshot_weights
    predicted_targets, (acc1, acc5) = accuracy(logits, target, topk=(1, 5))
    all_labels = predicted_targets if all_labels is None else torch.cat([all_labels, predicted_targets])
    top1 += acc1
    top5 += acc5

    n += images.size(0)

    #logits = 100. * image_features @ zeroshot_weights_
    #_, (acc1, acc5) = accuracy(logits, target, topk=(1, 5))
    #top1_ += acc1
    #top5_ += acc5
    #logits_mine = 100. * image_features @ zeroshot_weightss_
    #_, (acc1s, acc5s) = accuracy(logits_mine, target, topk=(1, 5))
    #top1s += acc1s
    #top5s += acc5s
    
cifar100.zs_targets = all_labels
top1 = (top1 / n) * 100
top5 = (top5 / n) * 100 
#top1_ = (top1_ / n) * 100
#top5_ = (top5_ / n) * 100 
#top1s = (top1s / n) * 100
#top5s = (top5s / n) * 100

print()
print(f"Top-1 accuracy: {top1:.2f}")# {top1_:.2f} {top1s:.2f}")
print(f"Top-5 accuracy: {top5:.2f}")# {top5_:.2f} {top5s:.2f}")

# %%
#da qui--- crea nuovo dataset con le immagini che non sono state classificate correttamente con etichetta sbagòliatas
#def accuracy(output, target, topk=(1,)):
#    pred = output.topk(max(topk), 1, True, True)[1].t()
#    correct = pred.eq(target.view(1, -1).expand_as(pred))
#    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

# %%
# %% PROBING
import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16",device)

@torch.no_grad()
def get_features(dataset):
    all_features = []
    all_labels = []

    for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
        features = model.encode_image(images.to(device))

        all_features.append(features)
        all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features
train_features, train_labels = get_features(cifar100)
test_features, test_labels = get_features(cifar100_test)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=500, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")

# %% INFERENCE
# %% INFERENCE (?)
torch.cuda.empty_cache()
import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16",device)

def get_features(dataset):
    all_features = []
    all_labels = []

    for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
        features = model.encode_image(images.to(device))

        all_features.append(features)
        all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features
train_features, train_labels = get_features(cifar100)

with torch.no_grad():
   test_features, test_labels = get_features(cifar100_test)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=500, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")

# %%
