from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Input
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers import BatchNormalization
from keras.regularizers import l2
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.optimizers import Adam
import pickle

file="disease(filled).csv"
df = pd.read_csv(path)

df

df.shape

df.isna().sum()

col_drop = ['Symptom_12', 'Symptom_13','Symptom_14','Symptom_15','Symptom_16','Symptom_17']
df = df.drop(columns=col_drop)

df

Disease = df['Disease'].unique()
Disease

len(Disease)

columns = df.columns[1:]
columns

symptoms = pd.unique(df[columns].values.ravel())
symptoms

len(symptoms)-1

label_encoder = LabelEncoder()
symp = label_encoder.fit_transform(symptoms)
symp

encodedSymp = dict(zip(symptoms, symp))
encodedSymp

for col in columns:
    df[col] = df[col].map(encodedSymp)

df

df['Disease'] = label_encoder.fit_transform(df['Disease'])

df

X = df.drop('Disease', axis=1)
y = df['Disease']

y

class_counts = y.value_counts()
print(class_counts)

y = to_categorical(y, num_classes=len(df['Disease'].unique()))

y.shape

X_train

X_test

y_test.shape

y_train.shape

model = Sequential()

model.add(Input(shape=(X_train.shape[1],)))

model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.02)))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.02)))
model.add(BatchNormalization())

model.add(Dense(y_train.shape[1], activation='softmax'))

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.summary()

history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes))

conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=df['Disease'].unique(), yticklabels=df['Disease'].unique())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

with open('disease_diagnosis_model_sequential.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)



