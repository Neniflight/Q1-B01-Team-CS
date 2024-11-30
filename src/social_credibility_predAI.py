
"""
NOTES:
1. Overall function of this file was creating and training a model for our social credibility predictive AI
    facutality factor. 
2. Keras didn't work out due to versions not compatible, and we've tried to switch to many different version
    but it still didn't work out, therefore, we've decided to redo this neural network in pytorch
"""
# imports 
import pandas as pd
import numpy as np
import sklearn
import tensorflow
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder

# load train data
train_data = pd.read_csv('https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/refs/heads/master/dataset/tsv/train2.tsv', sep = "\t")
first_data = train_data.columns
train_data.loc[train_data.shape[0]] = first_data

# data cleaning 
# (dropping na and only keeping 5 columns)
train_data.columns =['index','ID of statement', 'label', 'statement', 'subject', 'speaker', "speaker's job title", 'state info',
                     'party affiliation', 'barely true counts', 'false counts', 'half true counts', 'mostly true counts',
                    'pants on fire counts', 'context', 'extracted justification']
train_data = train_data.drop(columns=['index'])
train_data['subject'] = train_data['subject'].str.split(",")
train_data["speaker's job title"] = train_data["speaker's job title"].str.lower()
train_data["extracted justification"] = train_data["extracted justification"].str.split(" ")
train_data = train_data.dropna()
train_data = train_data[["label", "speaker", "context", "party affiliation"]]

# Ohe
ohe = OneHotEncoder(handle_unknown = "ignore", sparse_output = False)

# Ohe label
ohe_label = ohe.fit_transform(np.array(train_data['label']).reshape(-1,1))
train_data_ohe_label_df = pd.DataFrame(ohe_label, columns = list(train_data['label'].unique()))

# Ohe attributes speaker, context, and party affiliation
ohe_speaker = ohe.fit_transform(np.array(train_data['speaker']).reshape(-1,1))
ohe_context = ohe.fit_transform(np.array(train_data['context']).reshape(-1,1))
ohe_party = ohe.fit_transform(np.array(train_data['party affiliation']).reshape(-1,1))
ohe_speaker_ohe_context_ohe_party = []
for i in range(len(ohe_speaker)):
    ohe_speaker_ohe_context_ohe_party.append(np.concatenate((ohe_speaker[i], ohe_context[i], ohe_party[i])))
ohe_speaker_ohe_context_ohe_party = np.array(ohe_speaker_ohe_context_ohe_party)
small_training_ohe_speaker_context_party_df = pd.DataFrame(ohe_speaker_ohe_context_ohe_party, columns = 
                                             list(train_data['speaker'].unique()) 
                                             + list(train_data['context'].unique())
                                             + list(train_data['party affiliation'].unique()))
# citation: https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.OneHotEncoder.html

# keras simple nn
nn = keras.models.Sequential([
    keras.Input(shape = (small_training_ohe_speaker_context_party_df.shape[1],)),
    keras.layers.Dense(10, activation='sigmoid'),
    keras.layers.Dense(10, activation='sigmoid'),
    keras.layers.Dense(10, activation='sigmoid'),
    keras.layers.Dense(6),
])
# citation: https://keras.io/guides/sequential_model/

nn.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.01), loss = 'MeanSquaredError', metrics = ["accuracy"])

nn.fit(small_training_ohe_speaker_context_party_df, train_data_ohe_label_df, batch_size = 10, epochs = 100)


print("hello")
# save as h5
nn.save('social_cred_predAI.h5')
# citation: https://keras.io/getting_started/faq/