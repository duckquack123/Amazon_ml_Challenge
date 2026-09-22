# Data Preprocessing Pipeline
#
# This script processes the training data by:
# 1. Extracting Value and Unit from catalog_content
# 2. Processing image links for OCR text extraction
# 3. Applying NLP for feature engineering
# 4. Creating a clean, processed DataFrame for modeling

# -----------------
# Setup and Imports
# -----------------
import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
import requests
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
from io import BytesIO
import warnings
from tqdm import tqdm as tqdm_bar
import argparse
import concurrent.futures
from typing import List, Dict, Any, Optional
import uuid
import tempfile
from collections import Counter

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

TESSERACT_CMD = os.getenv('TESSERACT_CMD')
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def setup_environment() -> None:
    """
    Downloads required NLTK data.
    """
    print("Setting up the environment...")
    # Download required NLTK data
    for package in ['punkt', 'stopwords', 'wordnet']:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        except LookupError:
            nltk.download(package)

    print("Environment setup complete.")

# --------------------------------
# Data Extraction and Manipulation
# --------------------------------

def extract_value_unit(content: Optional[str]) -> pd.Series:
    """
    Extracts numerical value and unit from a string using regex.
    """
    if not isinstance(content, str):
        return pd.Series({'value': None, 'unit': ''})

    unit_patterns = {
        'kg': r'(?:kg|kgs|kilograms?)',
        'g': r'(?:g|grams?)',
        'l': r'(?:l|ltr|liters?|litres?)',
        'ml': r'(?:ml|milliliters?|millimetres?)',
        'cm': r'(?:cm|centimeters?|centimetres?)',
        'mm': r'(?:mm|millimeters?|millimetres?)',
        'in': r'(?:in|inch|inches)',
        'm': r'(?:m|meters?|metres?)',
        'oz': r'(?:oz|ounces?)',
        'lb': r'(?:lb|lbs|pounds?)'
    }

    unit_pattern = '|'.join(f'({p})' for p in unit_patterns.values())
    value_pattern = r'(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)'
    pattern = fr'{value_pattern}\s*({unit_pattern})'
    matches = re.findall(pattern, content, re.IGNORECASE)

    if matches:
        value_str, *units = matches[0]
        unit = next((u for u in units if u), '')

        if '-' in value_str:
            start, end = map(float, value_str.split('-'))
            value = (start + end) / 2
        else:
            value = float(value_str)

        unit = unit.lower().strip()
        for std_unit, patterns in unit_patterns.items():
            if re.match(f'^{patterns}$', unit, re.IGNORECASE):
                unit = std_unit
                break
        return pd.Series({'value': value, 'unit': unit})

    return pd.Series({'value': None, 'unit': ''})

def preprocess_image_for_ocr(img, max_size=800):
    # Resize image if too large
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    # Convert to grayscale
    img = ImageOps.grayscale(img)
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    return img

def extract_text_from_url(url: str) -> str:
    """
    Extracts text from an image URL using Tesseract OCR with preprocessing.
    """
    cache_dir = Path(__file__).resolve().parent / 'image_cache'
    os.makedirs(cache_dir, exist_ok=True)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        if 'image' not in response.headers.get('Content-Type', ''):
            print(f"URL did not return an image: {url}")
            return ''
        try:
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img = preprocess_image_for_ocr(img)
        except Exception as e:
            print(f"Invalid image for URL {url}: {e}")
            return ''
        try:
            text = pytesseract.image_to_string(img, lang='eng')
            text = re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            print(f"Tesseract OCR failed for {url}: {e}")
            text = ''
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for URL {url}: {e}")
    return ''

def process_images_in_parallel(df: pd.DataFrame, max_workers: int = 4) -> pd.Series:
    """
    Processes a DataFrame of image URLs in parallel to extract OCR text using Tesseract.
    """
    ocr_texts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(extract_text_from_url, url): url for url in df['image_link']}
        progress = tqdm_bar(concurrent.futures.as_completed(future_to_url), total=len(df), desc="Processing Images")
        results = {}
        for future in progress:
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception as exc:
                print(f'{url} generated an exception: {exc}')
                results[url] = ''
    return df['image_link'].map(results)

# -----------------
# Main Pipeline Steps
# -----------------

def load_data(file_path: Path) -> pd.DataFrame:
    """Loads data from a CSV file."""
    print(f"--- Step 1: Loading Data from {file_path} ---")
    if not file_path.exists():
        raise FileNotFoundError(f"Error: The file was not found at {file_path}")
    
    df = pd.read_csv(file_path)
    print("Dataset shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nSample data:\n", df.head())
    return df

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts value and unit from catalog_content."""
    print("\n--- Step 2: Extracting Value and Unit from 'catalog_content' ---")
    extracted = df['catalog_content'].apply(extract_value_unit)
    df[['value', 'unit']] = extracted
    print("Value and Unit extraction complete.")
    print("\nSample results:\n", df[['catalog_content', 'value', 'unit']].head())
    return df

def run_ocr_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts text from images using Tesseract OCR."""
    print("\n--- Step 3: Extracting Text from Images via Tesseract OCR ---")
    df['ocr_text'] = process_images_in_parallel(df)
    print("\nOCR extraction complete.")
    print("\nSample OCR results:\n", df[['image_link', 'ocr_text']].head())
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values in the DataFrame."""
    print("\n--- Step 4: Final Data Cleaning ---")
    print("Missing values before cleaning:\n", df.isnull().sum())
    
    df['value'] = df['value'].fillna(0)
    df['unit'] = df['unit'].fillna('')
    df['ocr_text'] = df['ocr_text'].fillna('')
    
    print("\nMissing values after cleaning:\n", df.isnull().sum())
    return df

def parse_item_name(content):
    if not isinstance(content, str):
        return ''
    match = re.search(r'Item Name: ([^\n]+)', content)
    if match:
        return match.group(1).strip()
    # Fallback: first line
    return content.split('\n')[0].strip()

def nlp_preprocessing(final_df):
    # Parse item_name
    final_df['item_name'] = final_df['catalog_content'].apply(parse_item_name) if 'catalog_content' in final_df.columns else ''
    # Combine item_name, OCR text
    def prepare_text(row):
        parts = []
        if 'item_name' in row and row['item_name']:
            parts.append(str(row['item_name']))
        if 'ocr_text' in row and row['ocr_text']:
            parts.append(str(row['ocr_text']))
        return ' '.join(parts)
    final_df['combined_text'] = final_df.apply(prepare_text, axis=1)
    # Tokenization and features
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        return tokens
    final_df['tokens'] = final_df['combined_text'].apply(preprocess_text)
    final_df['token_count'] = final_df['tokens'].apply(len)
    final_df['char_length'] = final_df['combined_text'].str.len().fillna(0).astype(int)
    final_df['top_tokens'] = final_df['tokens'].apply(lambda toks: Counter(toks).most_common(5))
    # For CSV, store tokens and top_tokens as strings
    final_df['tokens'] = final_df['tokens'].apply(str)
    final_df['top_tokens'] = final_df['top_tokens'].apply(str)
    return final_df

def save_data(df: pd.DataFrame, output_path: Path):
    """Saves the final DataFrame to a CSV file."""
    # Compose model-ready DataFrame
    columns = [
        'sample_id',
        'image_link',
        'ocr_text',
        'value',
        'unit',
    ]
    optional_columns = [
        'price',
        'item_name',
        'combined_text',
        'token_count',
        'char_length',
        'tokens',
        'top_tokens',
    ]
    model_df = df[[column for column in columns + optional_columns if column in df]].copy()
    model_df = nlp_preprocessing(model_df)
    print("\nFinal DataFrame shape:", model_df.shape)
    print("\nSample of final DataFrame:")
    print(model_df.head())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(output_path, index=False)
    print(f"\n✅ Successfully processed data saved to: {output_path}")

def main(args):
    """
    Main function to run the entire data processing pipeline.
    """
    try:
        setup_environment()
        
        df = load_data(args.input_file)
        df = extract_features(df)
        df = run_ocr_extraction(df)
        df = clean_data(df)
        save_data(df, args.output_file)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define project structure
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT_DIR / 'dataset'

    parser = argparse.ArgumentParser(description="Data preprocessing pipeline for Amazon ML Challenge.")
    parser.add_argument(
        "--input_file",
        type=Path,
        default=DATA_DIR / 'train.csv',
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=DATA_DIR / 'processed_for_model.csv',
        help="Path to save the processed CSV file."
    )
    
    args = parser.parse_args()
    main(args)
