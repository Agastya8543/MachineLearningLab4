import pandas as pd
import math

data = {
    'age': ['<=30', '<=30', '31-40', '>40', '>40', '>40', '31-40', '<=30', '<=30', '>40', '<=30', '31-40', '31-40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(data)

def calculate_entropy(df, attribute):
  unique_values = df[attribute].unique()
  probabilities = df[attribute].value_counts() / df.shape[0]
  entropy = 0.0
  for probability in probabilities:
    entropy += -probability * math.log2(probability)
  return entropy

entropy_per_attribute = {}
for attribute in df.columns[:-1]:
  entropy_per_attribute[attribute] = calculate_entropy(df, attribute)

def select_first_feature(entropy_per_attribute):

  max_entropy = max(entropy_per_attribute.values())
  first_feature = None
  for attribute, entropy in entropy_per_attribute.items():
    if entropy == max_entropy:
      first_feature = attribute
      break

  return first_feature

first_feature = select_first_feature(entropy_per_attribute)

print('Entropy for each attribute at the root node:')
for attribute in df.columns[:-1]:
  print(f"{attribute} = {entropy_per_attribute[attribute]}")

print('First feature to select for constructing the decision tree using Information Gain as the impurity measure:')
print(first_feature)
