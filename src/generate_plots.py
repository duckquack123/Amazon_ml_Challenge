#!/usr/bin/env python3
"""
Script to run EDA notebook cells and save plots with meaningful names for LaTeX document
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)

# Load and prepare text data
print("Loading data...")
train_df = pd.read_csv('../dataset/train.csv')

def extract_all_text(content):
    """Extract and clean all text from catalog content"""
    if not isinstance(content, str):
        return ''
    
    # Remove structured labels and keep only the content
    text = content.replace('Item Name:', '')
    text = re.sub(r'Bullet Point \d+:', '', text)
    text = text.replace('Value:', '').replace('Unit:', '')
    
    # Clean the text
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower().strip()
    
    return text

train_df['clean_text'] = train_df['catalog_content'].apply(extract_all_text)
text_mask = train_df['clean_text'].str.len() > 10
train_df_text = train_df[text_mask].copy()

print(f"Samples with meaningful text: {len(train_df_text)}")

# 1. WORD FREQUENCY ANALYSIS
print("Generating word frequency plots...")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

all_words = []
for text in train_df_text['clean_text']:
    tokens = word_tokenize(text)
    filtered_tokens = [
        lemmatizer.lemmatize(word) for word in tokens 
        if word not in stop_words and len(word) > 2
    ]
    all_words.extend(filtered_tokens)

word_freq = Counter(all_words)
top_20_words = word_freq.most_common(20)

# Plot 1: Word Frequency
plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
words, counts = zip(*top_20_words)
plt.barh(range(len(words)), counts, color='skyblue')
plt.yticks(range(len(words)), words)
plt.xlabel('Frequency')
plt.title('Top 20 Most Frequent Words')
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
all_counts = list(word_freq.values())
plt.hist(all_counts, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
plt.xlabel('Word Frequency')
plt.ylabel('Number of Words')
plt.title('Word Frequency Distribution')
plt.yscale('log')

plt.tight_layout()
plt.savefig('word_frequency_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. TF-IDF ANALYSIS
print("Generating TF-IDF plots...")
tfidf = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    min_df=5,
    max_df=0.95,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(train_df_text['clean_text'])
feature_names = tfidf.get_feature_names_out()
mean_tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)
tfidf_scores = list(zip(feature_names, mean_tfidf_scores))
tfidf_scores.sort(key=lambda x: x[1], reverse=True)
top_20_tfidf = tfidf_scores[:20]

plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
terms, scores = zip(*top_20_tfidf)
plt.barh(range(len(terms)), scores, color='coral')
plt.yticks(range(len(terms)), terms)
plt.xlabel('Mean TF-IDF Score')
plt.title('Top 20 TF-IDF Terms')
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
plt.hist(mean_tfidf_scores, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
plt.xlabel('TF-IDF Score')
plt.ylabel('Number of Terms')
plt.title('TF-IDF Score Distribution')

plt.tight_layout()
plt.savefig('tfidf_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. PRICE CORRELATION ANALYSIS
print("Generating price correlation plots...")
selector = SelectKBest(score_func=f_regression, k=20)
X_selected = selector.fit_transform(tfidf_matrix, train_df_text['price'])
selected_indices = selector.get_support(indices=True)
selected_features = feature_names[selected_indices]
feature_scores = selector.scores_[selected_indices]

price_correlated_terms = list(zip(selected_features, feature_scores))
price_correlated_terms.sort(key=lambda x: x[1], reverse=True)

plt.figure(figsize=(15, 6))
terms, scores = zip(*price_correlated_terms)
plt.barh(range(len(terms)), scores, color='gold')
plt.yticks(range(len(terms)), terms)
plt.xlabel('F-score (Correlation with Price)')
plt.title('Top 20 Terms Most Correlated with Price')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('price_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. CATEGORY ANALYSIS
print("Generating category plots...")
def extract_product_categories(text):
    categories = []
    food_terms = ['sauce', 'soup', 'chips', 'crackers', 'cookies', 'candy', 'chocolate', 
                  'cereal', 'snack', 'drink', 'juice', 'coffee', 'tea', 'spice', 'oil']
    health_terms = ['cream', 'lotion', 'shampoo', 'soap', 'vitamin', 'supplement', 
                    'medicine', 'care', 'beauty']
    household_terms = ['cleaner', 'detergent', 'paper', 'towel', 'bag', 'container',
                       'kitchen', 'bathroom']
    tech_terms = ['battery', 'charger', 'cable', 'tool', 'device', 'electronic']
    
    all_categories = {
        'Food & Beverage': food_terms,
        'Health & Beauty': health_terms,
        'Household': household_terms,
        'Electronics': tech_terms
    }
    
    text_lower = text.lower()
    found_categories = []
    
    for category, terms in all_categories.items():
        if any(term in text_lower for term in terms):
            found_categories.append(category)
    
    return found_categories if found_categories else ['Other']

train_df_text['categories'] = train_df_text['clean_text'].apply(extract_product_categories)
all_categories = []
for cat_list in train_df_text['categories']:
    all_categories.extend(cat_list)

category_counts = Counter(all_categories)
top_categories = category_counts.most_common(10)

plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
categories, counts = zip(*top_categories)
plt.barh(range(len(categories)), counts, color='lightgreen')
plt.yticks(range(len(categories)), categories)
plt.xlabel('Number of Products')
plt.title('Product Categories')
plt.gca().invert_yaxis()

# Price distribution by categories
plt.subplot(1, 2, 2)
top_5_categories = [cat for cat, _ in top_categories[:5]]
price_data = []
labels = []
for category in top_5_categories:
    category_mask = train_df_text['categories'].apply(lambda x: category in x)
    if category_mask.any():
        price_data.append(train_df_text[category_mask]['price'].values)
        labels.append(category)

plt.boxplot(price_data, labels=labels)
plt.title('Price Distribution by Category')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('category_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. BRAND AND DESCRIPTOR ANALYSIS
print("Generating brand and descriptor plots...")
def extract_brands(text):
    words = text.split()
    brands = []
    for word in words[:5]:
        if word.istitle() and len(word) > 2:
            brands.append(word.lower())
    return brands

def extract_descriptors(text):
    descriptors = []
    descriptor_patterns = [
        r'\b(organic|natural|premium|fresh|healthy|low|high|extra|super|mega|mini|large|small)\b',
        r'\b(creamy|crunchy|smooth|thick|thin|light|heavy|soft|hard)\b',
        r'\b(sweet|salty|spicy|hot|cold|warm|cool)\b',
        r'\b(new|original|classic|traditional|modern|vintage)\b'
    ]
    
    text_lower = text.lower()
    for pattern in descriptor_patterns:
        matches = re.findall(pattern, text_lower)
        descriptors.extend(matches)
    
    return descriptors

# Extract brands
all_brands = []
for text in train_df_text['catalog_content'].astype(str):
    item_match = re.search(r'Item Name: ([^\n]+)', text)
    if item_match:
        item_name = item_match.group(1)
        brands = extract_brands(item_name)
        all_brands.extend(brands)

brand_counts = Counter(all_brands)
top_20_brands = brand_counts.most_common(20)

# Extract descriptors
all_descriptors = []
for text in train_df_text['clean_text']:
    descriptors = extract_descriptors(text)
    all_descriptors.extend(descriptors)

descriptor_counts = Counter(all_descriptors)
top_20_descriptors = descriptor_counts.most_common(20)

plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
brands, brand_counts_vals = zip(*top_20_brands)
plt.barh(range(len(brands)), brand_counts_vals, color='purple', alpha=0.7)
plt.yticks(range(len(brands)), brands)
plt.xlabel('Mentions')
plt.title('Top 20 Brands')
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
descriptors, desc_counts_vals = zip(*top_20_descriptors)
plt.barh(range(len(descriptors)), desc_counts_vals, color='orange', alpha=0.7)
plt.yticks(range(len(descriptors)), descriptors)
plt.xlabel('Mentions')
plt.title('Top 20 Product Descriptors')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('brand_descriptor_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("All plots saved successfully!")
print("Generated files:")
print("- word_frequency_analysis.png")
print("- tfidf_analysis.png") 
print("- price_correlation_analysis.png")
print("- category_analysis.png")
print("- brand_descriptor_analysis.png")