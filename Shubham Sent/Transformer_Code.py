# -*- coding: utf-8 -*-
"""Multimodal Price Prediction (with Log-Price Training + 95th Percentile Cap + SMAPE Evaluation)"""

# ==============================================================================
# SEGMENT 1: SETUP AND INSTALLATIONS
# ==============================================================================
# !pip install transformers torch accelerate -q

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
import os
from pathlib import Path
from termcolor import colored  # For colorful console output

warnings.filterwarnings('ignore')

# ==============================================================================
# SEGMENT 2: CONFIGURATION AND DATA LOADING
# ==============================================================================

class Config:
    ROOT_DIR = Path(__file__).resolve().parents[1]
    TRAIN_FILE = os.getenv('TRAIN_FILE', str(ROOT_DIR / 'dataset2' / 'train_embeddings.csv'))
    TEST_FILE = os.getenv('TEST_FILE', str(ROOT_DIR / 'dataset2' / 'test_embeddings.csv'))
    TRAIN_EMBEDDINGS_FILE = os.getenv(
        'TRAIN_EMBEDDINGS_FILE', str(ROOT_DIR / 'embeddings' / 'train_embeddings.npz')
    )
    TEST_EMBEDDINGS_FILE = os.getenv(
        'TEST_EMBEDDINGS_FILE', str(ROOT_DIR / 'embeddings' / 'test_embeddings.npz')
    )
    TEXT_MODEL = 'distilbert-base-uncased'
    MAX_TEXT_LENGTH = 256
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    VAL_SPLIT_SIZE = 0.1

# --- Load CSV and Embeddings ---
required_files = [
    Config.TRAIN_FILE, Config.TEST_FILE,
    Config.TRAIN_EMBEDDINGS_FILE, Config.TEST_EMBEDDINGS_FILE
]

if not all(os.path.exists(f) for f in required_files):
    print("Please make sure all required files are uploaded:")
    for f in required_files:
        if not os.path.exists(f):
            print(f" - Missing: {f}")
else:
    print("Loading data and embeddings...")
    train_df_full = pd.read_csv(Config.TRAIN_FILE).drop('embedding', axis=1, errors='ignore')
    test_df = pd.read_csv(Config.TEST_FILE).drop('embedding', axis=1, errors='ignore')

    with np.load(Config.TRAIN_EMBEDDINGS_FILE) as data:
        train_image_embeddings_full = data['embeddings']
    with np.load(Config.TEST_EMBEDDINGS_FILE) as data:
        test_image_embeddings = data['embeddings']

    # Split train and validation
    train_df, val_df, train_image_embeddings, val_image_embeddings = train_test_split(
        train_df_full, train_image_embeddings_full, test_size=Config.VAL_SPLIT_SIZE, random_state=42
    )

    # --- Cap prices at 95th percentile ---
    cap_value = train_df['price'].quantile(0.95)
    train_df['price'] = np.minimum(train_df['price'], cap_value)
    val_df['price'] = np.minimum(val_df['price'], cap_value)

    print(colored(f"95th percentile price cap: {cap_value:.2f}", 'cyan'))
    print(colored(f"New mean price: {train_df['price'].mean():.2f} | Median price: {train_df['price'].median():.2f}", 'yellow'))
    print(colored(f"Data loaded. Training: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}", 'green'))

text_tokenizer = DistilBertTokenizer.from_pretrained(Config.TEXT_MODEL)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(colored(f"Using device: {device}", 'magenta'))

# ==============================================================================
# SEGMENT 3: CUSTOM DATASET
# ==============================================================================
class ProductDataset(Dataset):
    def __init__(self, df, tokenizer, image_embeddings, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.image_embeddings = image_embeddings
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['catalog_content'])
        image_embedding = self.image_embeddings[idx]

        text_inputs = self.tokenizer(
            text,
            max_length=Config.MAX_TEXT_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'image_embedding': torch.tensor(image_embedding, dtype=torch.float)
        }

        if not self.is_test:
            price = np.log1p(row['price'])  # log(1 + price)
            item['price'] = torch.tensor(price, dtype=torch.float)
        return item

# ==============================================================================
# SEGMENT 4: MODEL
# ==============================================================================
class MultiModalPricer(nn.Module):
    def __init__(self, image_embedding_dim):
        super().__init__()
        self.text_tower = DistilBertModel.from_pretrained(Config.TEXT_MODEL)
        combined_dim = self.text_tower.config.dim + image_embedding_dim

        self.regressor = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, input_ids, attention_mask, image_embedding):
        text_output = self.text_tower(input_ids=input_ids, attention_mask=attention_mask)
        text_embed = text_output.last_hidden_state[:, 0, :]
        combined_features = torch.cat([text_embed, image_embedding], dim=1)
        price_pred = self.regressor(combined_features)
        return price_pred

# ==============================================================================
# SEGMENT 5: SMAPE FUNCTION
# ==============================================================================
def smape_loss(y_pred, y_true):
    numerator = torch.abs(y_pred - y_true)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)).clamp(min=1e-8) / 2
    return torch.mean(numerator / denominator) * 100

# ==============================================================================
# SEGMENT 6: TRAINING AND VALIDATION
# ==============================================================================
train_dataset = ProductDataset(train_df, text_tokenizer, train_image_embeddings)
val_dataset = ProductDataset(val_df, text_tokenizer, val_image_embeddings)
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

image_embedding_dim = train_image_embeddings.shape[1]
model = MultiModalPricer(image_embedding_dim=image_embedding_dim).to(device)
optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
loss_fn = nn.MSELoss()

print(colored("\n🚀 Starting model training...\n", 'green'))

for epoch in range(Config.EPOCHS):
    model.train()
    total_train_loss, total_train_smape = 0, 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [Training]")

    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image_embedding = batch['image_embedding'].to(device)
        prices = batch['price'].to(device)

        predictions = model(input_ids, attention_mask, image_embedding).squeeze()
        loss = loss_fn(predictions, prices)
        loss.backward()
        optimizer.step()

        # SMAPE on original scale
        smape = smape_loss(torch.expm1(predictions), torch.expm1(prices))
        total_train_loss += loss.item()
        total_train_smape += smape.item()
        progress_bar.set_postfix({
            'Train MSE': f"{loss.item():.4f}",
            'Train SMAPE': f"{smape.item():.2f}"
        })

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_smape = total_train_smape / len(train_loader)
    print(colored(f"\nEpoch {epoch + 1} ✅ Training MSE: {avg_train_loss:.4f} | SMAPE: {avg_train_smape:.2f}", 'cyan'))

    # Validation
    model.eval()
    total_val_loss, total_val_smape = 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [Validation]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_embedding = batch['image_embedding'].to(device)
            prices = batch['price'].to(device)

            predictions = model(input_ids, attention_mask, image_embedding).squeeze()
            val_loss = loss_fn(predictions, prices)
            val_smape = smape_loss(torch.expm1(predictions), torch.expm1(prices))

            total_val_loss += val_loss.item()
            total_val_smape += val_smape.item()

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_smape = total_val_smape / len(val_loader)
    print(colored(f"Epoch {epoch + 1} 🧾 Validation MSE: {avg_val_loss:.4f} | SMAPE: {avg_val_smape:.2f}\n", 'yellow'))

print(colored("\n✅ Training complete!", 'green'))

# ==============================================================================
# SEGMENT 7: INFERENCE
# ==============================================================================
print(colored("\n🔮 Generating predictions for the test set...", 'cyan'))

test_dataset = ProductDataset(test_df, text_tokenizer, test_image_embeddings, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

all_predictions = []
model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image_embedding = batch['image_embedding'].to(device)

        predictions = model(input_ids, attention_mask, image_embedding)
        preds = torch.expm1(predictions.squeeze())  # reverse log1p
        all_predictions.extend(preds.cpu().numpy())

final_predictions = np.maximum(np.array(all_predictions), 0.01)

submission_df = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': final_predictions
})
submission_df.to_csv('submission.csv', index=False)

print(colored("\n💾 Submission file 'submission.csv' generated successfully!", 'green'))
print(submission_df.head())


# # -*- coding: utf-8 -*-
# """Untitled0.ipynb

# Automatically generated by Colab.

# Original file is located at
#     https://colab.research.google.com/drive/1VsuQiksiCyYr1nCxASBSwdC-WJt7Mf5i
# """



# # ==============================================================================
# # SEGMENT 1: SETUP AND INSTALLATIONS
# # ==============================================================================
# # Run this command in a separate cell first!
# # !pip install transformers torch accelerate -q

# import pandas as pd
# import numpy as np
# import torch
# from torch import nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import DistilBertTokenizer, DistilBertModel
# from torch.optim import AdamW  # Corrected import for AdamW
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# import warnings
# import os

# warnings.filterwarnings('ignore')

# # ==============================================================================
# # SEGMENT 2: CONFIGURATION AND DATA LOADING
# # ==============================================================================

# # --- Configuration ---
# class Config:
#     TRAIN_FILE = 'dataset2/train_embeddings.csv'
#     TEST_FILE = 'dataset2/test_embeddings.csv'
#     TRAIN_EMBEDDINGS_FILE = 'embeddings/train_embeddings.npz'
#     TEST_EMBEDDINGS_FILE = 'embeddings/test_embeddings.npz'
#     TEXT_MODEL = 'distilbert-base-uncased'
#     MAX_TEXT_LENGTH = 256
#     BATCH_SIZE = 32
#     EPOCHS = 3 # Recommend 3-5 epochs
#     LEARNING_RATE = 2e-5
#     VAL_SPLIT_SIZE = 0.1

# # --- Check for files and load data ---
# required_files = [Config.TRAIN_FILE, Config.TEST_FILE, Config.TRAIN_EMBEDDINGS_FILE, Config.TEST_EMBEDDINGS_FILE]
# if not all(os.path.exists(f) for f in required_files):
#     print("Please make sure all required files are uploaded:")
#     for f in required_files:
#         if not os.path.exists(f):
#             print(f" - Missing: {f}")
# else:
#     print("Loading data and embeddings...")
#     train_df_full = pd.read_csv(Config.TRAIN_FILE)
#     train_df_full = train_df_full.drop('embedding', axis=1)  # Drop the old embeddings column if present
#     test_df = pd.read_csv(Config.TEST_FILE)
#     test_df = test_df.drop('embedding', axis=1)  # Drop the old embeddings column if present   


#     # Load image embeddings
#     with np.load(Config.TRAIN_EMBEDDINGS_FILE) as data:
#         train_image_embeddings_full = data['embeddings']
#     with np.load(Config.TEST_EMBEDDINGS_FILE) as data:
#         test_image_embeddings = data['embeddings']

#     # Split training data and embeddings for validation, ensuring they stay aligned
#     train_df, val_df, train_image_embeddings, val_image_embeddings = train_test_split(
#         train_df_full,
#         train_image_embeddings_full,
#         test_size=Config.VAL_SPLIT_SIZE,
#         random_state=42
#     )
#     print(f"Data loaded. Training samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")

# # --- Initialize Tokenizer ---
# text_tokenizer = DistilBertTokenizer.from_pretrained(Config.TEXT_MODEL)

# # Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# # ==============================================================================
# # SEGMENT 3: CUSTOM PYTORCH DATASET (MODIFIED FOR NPZ)
# # ==============================================================================
# class ProductDataset(Dataset):
#     def __init__(self, df, tokenizer, image_embeddings, is_test=False):
#         self.df = df
#         self.tokenizer = tokenizer
#         self.image_embeddings = image_embeddings
#         self.is_test = is_test

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         text = str(row['catalog_content'])
#         image_embedding = self.image_embeddings[idx]

#         # Process text
#         text_inputs = self.tokenizer(
#             text,
#             max_length=Config.MAX_TEXT_LENGTH,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )

#         item = {
#             'input_ids': text_inputs['input_ids'].squeeze(0),
#             'attention_mask': text_inputs['attention_mask'].squeeze(0),
#             'image_embedding': torch.tensor(image_embedding, dtype=torch.float)
#         }

#         if not self.is_test:
#             item['price'] = torch.tensor(row['price'], dtype=torch.float)

#         return item

# # ==============================================================================
# # SEGMENT 4: MULTI-MODAL MODEL ARCHITECTURE (MODIFIED FOR NPZ)
# # ==============================================================================
# class MultiModalPricer(nn.Module):
#     def __init__(self, image_embedding_dim):
#         super().__init__()
#         self.text_tower = DistilBertModel.from_pretrained(Config.TEXT_MODEL)

#         # The image tower is removed. We use the dimension of the pre-computed embeddings.
#         combined_dim = self.text_tower.config.dim + image_embedding_dim

#         self.regressor = nn.Sequential(
#             nn.LayerNorm(combined_dim),
#             nn.Linear(combined_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 1)
#         )

#     def forward(self, input_ids, attention_mask, image_embedding):
#         # Text tower forward pass
#         text_output = self.text_tower(input_ids=input_ids, attention_mask=attention_mask)
#         text_embed = text_output.last_hidden_state[:, 0, :]

#         # No image tower needed. Concatenate directly.
#         combined_features = torch.cat([text_embed, image_embedding], dim=1)
#         price_pred = self.regressor(combined_features)
#         return price_pred

# # ==============================================================================
# # SEGMENT 5: TRAINING AND VALIDATION
# # ==============================================================================

# # Create Datasets and DataLoaders
# train_dataset = ProductDataset(train_df, text_tokenizer, train_image_embeddings)
# val_dataset = ProductDataset(val_df, text_tokenizer, val_image_embeddings)

# train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
# val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

# # Initialize Model, Optimizer, and Loss Function
# # The model needs to know the dimension of your image embeddings
# image_embedding_dim = train_image_embeddings.shape[1]
# model = MultiModalPricer(image_embedding_dim=image_embedding_dim).to(device)
# optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
# loss_fn = nn.MSELoss() # We'll use RMSE for reporting

# print("\nStarting model training...")
# for epoch in range(Config.EPOCHS):
#     model.train()
#     total_train_loss = 0
#     progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [Training]")

#     for batch in progress_bar:
#         optimizer.zero_grad()

#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         image_embedding = batch['image_embedding'].to(device)
#         prices = batch['price'].to(device)

#         predictions = model(input_ids, attention_mask, image_embedding)

#         loss = torch.sqrt(loss_fn(predictions.squeeze(), prices)) # RMSE
#         total_train_loss += loss.item()

#         loss.backward()
#         optimizer.step()
#         progress_bar.set_postfix({'Train RMSE': loss.item()})

#     avg_train_loss = total_train_loss / len(train_loader)
#     print(f"Epoch {epoch + 1} - Average Training RMSE Loss: {avg_train_loss:.4f}")

#     # Validation loop
#     model.eval()
#     total_val_loss = 0
#     with torch.no_grad():
#         for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [Validation]"):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             image_embedding = batch['image_embedding'].to(device)
#             prices = batch['price'].to(device)

#             predictions = model(input_ids, attention_mask, image_embedding)
#             loss = torch.sqrt(loss_fn(predictions.squeeze(), prices))
#             total_val_loss += loss.item()

#     avg_val_loss = total_val_loss / len(val_loader)
#     print(f"Epoch {epoch + 1} - Average Validation RMSE Loss: {avg_val_loss:.4f}")

# print("\nTraining complete!")

# # ==============================================================================
# # SEGMENT 6: INFERENCE AND SUBMISSION FILE GENERATION
# # ==============================================================================

# print("\nGenerating predictions for the test set...")

# # Create test dataset and dataloader
# test_dataset = ProductDataset(test_df, text_tokenizer, test_image_embeddings, is_test=True)
# test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

# all_predictions = []
# model.eval()
# with torch.no_grad():
#     for batch in tqdm(test_loader, desc="Predicting"):
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         image_embedding = batch['image_embedding'].to(device)

#         predictions = model(input_ids, attention_mask, image_embedding)
#         all_predictions.extend(predictions.squeeze().cpu().numpy())

# # Ensure predictions are positive
# final_predictions = np.maximum(np.array(all_predictions), 0.01)

# # Create submission DataFrame
# submission_df = pd.DataFrame({
#     'sample_id': test_df['sample_id'],
#     'price': final_predictions
# })

# # Save to CSV
# submission_df.to_csv('submission.csv', index=False)

# print("\nSubmission file 'submission.csv' has been generated successfully!")
# print("Top 5 predictions:")
# print(submission_df.head())