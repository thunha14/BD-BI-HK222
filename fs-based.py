import recordlinkage
import pandas as pd
from recordlinkage import classifiers

df1 = pd.read_csv('./datasets/DBLP2.csv')
df2 = pd.read_csv('./datasets/ACM.csv')

# Blocking
indexer = recordlinkage.Index()
indexer.block('year')
candidate_pairs = indexer.index(df1, df2)

compare = recordlinkage.Compare()
compare.string("title", "title", label="title")
compare.string("authors", "authors", method="levenshtein", label="authors")
compare.string("venue", "venue", method="levenshtein", label="venue")
features = compare.compute(candidate_pairs, df1, df2)

# Train the ECM classifier
ecm = recordlinkage.KMeansClassifier()
features_vec = ecm.fit_predict(features)

pred_matched = set()
for i, j in features_vec:
    rec1 = df1['id'][i]
    rec2 = df2['id'][j]
    pred_matched.add((rec1, rec2))

df = pd.DataFrame(pred_matched, columns=['idDBLP', 'idACM'])
df.to_csv('./datasets/pred_matched.csv', index=False)
# print(matched_pairs)

true_df = pd.read_csv('./datasets/DBLP-ACM_perfectMapping.csv')

true_matched = list(zip(true_df['idDBLP'], true_df['idACM']))

# Calculate the performance metrics
inter = pred_matched.intersection(true_matched)
precision = len(inter) / len(pred_matched)
recall = len(inter) / len(true_matched)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1_score}')
