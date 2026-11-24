import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

df = pd.DataFrame({
    'age': [25, None, 40, 35, None],
    'income': [30000, 45000, None, 80000, 52000],
    'city': ['Chennai', 'Mumbai', None, 'Chennai', 'Delhi'],
    'size': ['S', 'L', 'M', 'S', 'L']
})

print("Original DataFrame:")
print(df)

simple_df = df.copy()

numeric_columns = ['age', 'income']

simple_imputer = SimpleImputer(strategy='median')

simple_df[numeric_columns] = simple_imputer.fit_transform(
    simple_df[numeric_columns]
)

print("\nAfter SimpleImputer (Median):")
print(simple_df)

city_imputer = SimpleImputer(strategy='most_frequent')

simple_df[['city']] = city_imputer.fit_transform(
    simple_df[['city']]
)

print("\nAfter SimpleImputer (Most Frequent City):")
print(simple_df)

knn_df = df.copy()

knn_imputer = KNNImputer(n_neighbors=2)

knn_df[numeric_columns] = knn_imputer.fit_transform(
    knn_df[numeric_columns]
)

print("\nAfter KNNImputer:")
print(knn_df[numeric_columns])

comparison = pd.DataFrame({
    'SimpleImputer Age': simple_df['age'],
    'KNNImputer Age': knn_df['age'],
    'SimpleImputer Income': simple_df['income'],
    'KNNImputer Income': knn_df['income']
})

print("\nSimpleImputer vs KNNImputer Comparison:")
print(comparison)

onehot_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

city_encoded = onehot_encoder.fit_transform(
    simple_df[['city']]
)

city_columns = onehot_encoder.get_feature_names_out(['city'])

city_encoded_df = pd.DataFrame(
    city_encoded,
    columns=city_columns,
    index=simple_df.index
)

print("\nAfter OneHotEncoding City:")
print(city_encoded_df)

ordinal_encoder = OrdinalEncoder(
    categories=[['S', 'M', 'L']]
)

simple_df['size'] = ordinal_encoder.fit_transform(
    simple_df[['size']]
).astype(int)

print("\nAfter OrdinalEncoding Size:")
print(simple_df[['size']])


final_df = pd.concat(
    [
        simple_df[['age', 'income']],
        city_encoded_df,
        simple_df[['size']]
    ],
    axis=1
)

print("\nFinal Encoded DataFrame:")
print(final_df)


# LabelEncoder - Target column
# The provided DataFrame does not contain a target column.
# LabelEncoder should be applied to the actual target labels,
# not to input features.
# We need the actual target column before applying it.


# Explanation
# LabelEncoder converts target class labels into numbers.
# It should not be used for input features because the assigned
# numbers may create a false numerical order between categories.