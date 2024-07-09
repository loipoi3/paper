import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from code.experiments.binary_classification_image_data.model import Model
from custom_dataset import ChestXRayDataset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = ChestXRayDataset("train")
    test_dataset = ChestXRayDataset("test")
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    model = Model(3, True)
    checkpoint = torch.load(
        'experiments/multiclass_classification_image_data/binary_classification_image_data_model.pth')
    model.load_state_dict(checkpoint)
    model = model.to(device)

    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            batch_embeddings = model(images)
            batch_embeddings = batch_embeddings.view(batch_embeddings.size(0), -1)
            embeddings_list.append(batch_embeddings.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    all_labels = np.array(labels_list).reshape(-1, 1)
    final_data = np.hstack((all_embeddings, all_labels))
    column_names = [f'feature_{i + 1}' for i in range(final_data.shape[1] - 1)] + ['label']
    df = pd.DataFrame(final_data, columns=column_names)
    df.to_csv('datasets/chest_xray/train_embeddings_multiclass.csv', index=False)
    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            batch_embeddings = model(images)
            batch_embeddings = batch_embeddings.view(batch_embeddings.size(0), -1)
            embeddings_list.append(batch_embeddings.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    all_labels = np.array(labels_list).reshape(-1, 1)
    final_data = np.hstack((all_embeddings, all_labels))
    column_names = [f'feature_{i + 1}' for i in range(final_data.shape[1] - 1)] + ['label']
    df = pd.DataFrame(final_data, columns=column_names)
    df.to_csv('datasets/chest_xray/test_embeddings_multiclass.csv', index=False)
