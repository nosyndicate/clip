import pandas as pd
import torch
from transformers import DistilBertTokenizer

from config import CFG
from model import CLIPModel
from dataset import get_transforms
import cv2
import os
import numpy as np


def compute_accuracy(similarity, k, count):
    top_k_indices = np.argsort(similarity, axis=1)[:, -k:]
    correct_indices = np.arange(count).reshape(-1, 1)
    matches_top_k = np.any(top_k_indices == correct_indices, axis=1)
    print(matches_top_k.mean())
    


def main(seed: int) -> None:
    # load model
    model = CLIPModel().cuda()
    checkpoint = torch.load(
        "best.pt",
        weights_only=True,
    )
    model.load_state_dict(checkpoint)
    model.eval()

    # load dataset
    df = pd.read_csv("data/captions.txt")
    random_sample = df.sample(n=500, random_state=seed)
    image_names = random_sample["image"].tolist()
    captions = random_sample["caption"].tolist()
    # print("========Captions=========")
    # print(captions)
    # print("========Captions=========")

    # get text embeddings
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    tokens = tokenizer(
        captions, padding=True, truncation=True, max_length=CFG.max_length
    )
    items = {key: torch.tensor(values).cuda() for key, values in tokens.items()}

    with torch.no_grad():
        text_embeddings = model.text_encoder(**items)
        text_embeddings = model.text_projection(text_embeddings)

    # get image embeddings
    transforms = get_transforms("eval")
    images = []
    for image_name in image_names:
        image = cv2.imread(os.path.join("data/Images", image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms(image=image)["image"]
        image_tensor = torch.tensor(image).permute(2, 0, 1).float()
        images.append(image_tensor)

    image_batch = torch.stack(images).cuda()
    with torch.no_grad():
        image_embeddings = model.image_encoder(image_batch)
        image_embeddings = model.image_projection(image_embeddings)

    print(
        f"text embeddings shape is {text_embeddings.shape}, image embeddings shape is {image_embeddings.shape}"
    )

    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    similarity = text_embeddings.cpu().numpy() @ image_embeddings.cpu().numpy().T

    compute_accuracy(similarity, 1, 500)
    compute_accuracy(similarity, 3, 500)


if __name__ == "__main__":
    seed = 42
    main(seed)
