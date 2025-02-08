import pickle
import numpy as np
from RandomForest import RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data
try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))
    data = np.asarray(data_dict.get('data', []))
    labels = np.asarray(data_dict.get('labels', []))
except (FileNotFoundError, EOFError, KeyError) as e:
    print(f"Error loading data: {e}")
    exit()

if data.size == 0 or labels.size == 0:
    print("Error: No data found in data.pickle. Ensure the file contains valid data.")
    exit()

print(f'Data loaded: {data.shape[0]} samples with {data.shape[1] if data.ndim > 1 else 0} features each')

# Convert string labels to numeric labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
print(f'Training data: {x_train.shape[0]} samples')
print(f'Testing data: {x_test.shape[0]} samples')

# Initialize and train the Random Forest model
model = RandomForest(n_trees=10, max_depth=10, min_samples_split=2)
model.fit(x_train, y_train)
print('Training complete.')

# Predict on test data
y_predict = model.predict(x_test)

# Evaluate accuracy
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': label_encoder}, f)

# import pickle

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np


# data_dict = pickle.load(open('./data.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# model = RandomForestClassifier()

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly !'.format(score * 100))

# f = open('model.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()