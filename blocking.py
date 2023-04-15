import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(data):
    # Preprocessing steps like lowercasing, removing special characters, etc.
    return data

def find_key_features(data, n_features=10):
    # Preprocess the data
    preprocessed_data = preprocess_data(data)

    # Compute term frequency
    counter = Counter()
    for record in preprocessed_data:
        counter.update(record.split())

    # Extract the n_features most common terms
    key_features = [feature for feature, _ in counter.most_common(n_features)]
    return key_features

def block_records(data, key_features):
    blocks = {}
    for record in data:
        for feature in key_features:
            if feature in record:
                if feature not in blocks:
                    blocks[feature] = []
                blocks[feature].append(record)
                break
    return blocks

def main():
    # Load your data
    df1 = pd.read_csv('./datasets/DBLP2.csv')
    df2 = pd.read_csv('./datasets/ACM.csv')

    # Find key features
    key_features = find_key_features(df2)

    # Block records
    blocked_data = block_records(df2, key_features)
    
    print("Blocked Data:", blocked_data)

if __name__ == "__main__":
    main()