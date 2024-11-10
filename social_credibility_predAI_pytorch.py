# imports
import pandas as pd
import numpy as np
# import scipy
import sklearn
from sklearn.preprocessing import OneHotEncoder

# EDA
train_data = pd.read_csv('https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/refs/heads/master/dataset/tsv/train2.tsv', sep = "\t")
train_data.columns =['index','ID of statement', 'label', 'statement', 'subject', 'speaker', "speaker's job title", 'state info',
                     'party affiliation', 'barely true counts', 'false counts', 'half true counts', 'mostly true counts',
                    'pants on fire counts', 'context', 'extracted justification']
train_data = train_data[["label", "speaker", "context", "party affiliation"]]
train_data = train_data.dropna()
modified_label = train_data.copy()
modified_label['mod_label'] = modified_label['label'].replace({'pants-fire': 0, np.nan : 0, 'barely-true':2, 'false':4, 'half-true':6, 'mostly-true':8, 'true':10})
modified_label.head()

# clean up context column, get rid of all useless words like "a", "the", etc, and narrow down the categories of context
# write function to clean context
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
unique_characters = '!@#$%^&*()_+~{}|:"<>?,./;'

# light cleaning (fixing syntax)
def clean_context(context):
  context = context.split(" ")
  cleaned_context = []
  for w in context:
    cleaned_word = ""
    if w not in stopwords.words('english'):
      for char in w:
        if char not in unique_characters:
          cleaned_word += char
      cleaned_context.append(cleaned_word)
  return str.lower(" ".join(cleaned_context))

# deep cleaning (fixing to reduce category)
def clean_context_by_cat (context):
  context = context.split(" ")
  for w in context:
    if w == "press":
      output = "press"
      break
    elif w == "news" or w == "abc's" or w == "msnbc's" or w =="nbc's" or w == "journal-constitution" or w == "reporters" or w == "providence" or w == "amanpour":
      output = "news"
      break
    elif w == "newspaper" or w == "study" or w == "chart":
      output = "facts_approved_text"
      break
    elif w == "speech" or w =="speeches":
      output = "speech"
      break
    elif w == "interview" or w == "interviews" or w == "appearance":
      output = "interview"
      break
    elif w == "ad" or w == "advertisement" or w =="commercial" or w == "flier" or w == "fliers":
      output = "ads"
      break
    elif w == "debate":
      output = "debate"
      break
    elif w == "conference":
      output = "conference"
      break
    elif w == "campaign" or w == "candidate" or w == "rally" or w =="bill-signing" or w == "cnn's":
      output = "campaign"
      break
    elif w == "statement" or w == "statements":
      output = "statement"
      break
    elif w == "web" or w == "website" or w =="websites" or w == "internet" or w == "internets" or w == "blogs":
      output = "web"
      break
    elif w == "meeting" or w == "meetings":
      output = "meeting"
      break
    elif w == "email" or w == "e-mail" or w == "mailer" or w == "letter" or w =="letters" or w == "mailing":
      output = "mail"
      break
    elif w == "petition":
      output = "petition"
      break
    elif w == "comments" or w == "comment":
      output = "comments"
      break
    elif w == "op-ed" or w == "opinion" or w == "column" or w == "editor" or w == "editorial" or w == "commentary" or w == "forum" or w == "panel" or w == "discussion" or w == "radio" or w == "town" or w == "questionnaire" or w =="o'reilly" or w == "survey":
      output = "opinion piece"
      break
    elif w == "address" or w == "union":
      output = "address"
      break
    elif w == "social_media" or w == "tweet" or w == "tweets" or w == "facebook" or w == "blog" or w == "posts" or w == "posting" or w == "post" or w == "twitter":
      output = "social media"
      break
    elif w == "broadcast":
      output = "broadcast"
      break
    elif w == "video":
      output = "video"
      break
    elif w == "article" or w =="newsletter":
      output = "article"
      break
    elif w == "report" or w =="presenation":
      output = "presentation"
      break
    elif w == "hearing":
      output = "hearing"
      break
    elif w == "remarks":
      output = "remarks"
      break
    elif w == "book" or w == "books" or w == "movie" or w == "billboard" or w == "show" or w == "episode" or w == "comics":
      output = "entertainment"
      break
    elif w =="meme" or w == "rounds" or w == "forwarded":
      output = "meme"
      break
    elif w == "oxford":
      output = "oxford"
      break
    elif w == "fla" or w == "boca":
      output = "florida"
      break
    elif w == "minn":
      output = "minnesota"
      break
    elif w == "iowa":
      output = "iowa"
      break
    elif w == "mo" or w == "louis":
      output = "missouri"
      break
    elif w == "mich":
      output = "michigan"
      break
    elif w == "pa":
      output = "Pennsylvania"
      break
    elif w == "vegas" or w =="nev":
      output = "nevada"
      break
    elif w =="tenn":
      output = "Tennessee"
      break
    elif w == "nh":
      output = "New Hampshire"
      break
    elif w == "colo":
      output = "colorado"
      break
    else:
      output = ""
  return output

# apply function back to df['context']
modified_label['context'] = modified_label['context'].apply(clean_context)
modified_label['context'] = modified_label['context'].apply(clean_context_by_cat)
modified_label.head(50)

# drop the empty outputs, weird/unique context
modified_label = modified_label[modified_label['context'] != ""]
modified_label.head()

# citation: https://www.geeksforgeeks.org/removing-stop-words-nltk-python/

# one-hot-encoding
ohe = OneHotEncoder(sparse_output = False)

ohe_label = ohe.fit_transform(np.array(modified_label['label']).reshape(-1,1))
ohe_label_df = pd.DataFrame(ohe_label, columns = list(modified_label['label'].unique()))

ohe_speaker = ohe.fit_transform(np.array(modified_label['speaker']).reshape(-1,1))
ohe_context = ohe.fit_transform(np.array(modified_label['context']).reshape(-1,1))
ohe_party = ohe.fit_transform(np.array(modified_label['party affiliation']).reshape(-1,1))
ohe_speaker_ohe_context_ohe_party = []
for i in range(len(ohe_speaker)):
    ohe_speaker_ohe_context_ohe_party.append(np.concatenate((ohe_speaker[i], ohe_context[i], ohe_party[i])))
ohe_speaker_ohe_context_ohe_party = np.array(ohe_speaker_ohe_context_ohe_party)


ohe_speaker_context_party_df = pd.DataFrame(ohe_speaker_ohe_context_ohe_party, columns = 
                                             list(modified_label['speaker'].unique()) 
                                             + list(modified_label['context'].unique())
                                                + list(modified_label['party affiliation'].unique()))


modified_data = ohe_speaker_context_party_df[0: int(ohe_speaker_context_party_df.shape[0]*0.8)]
training_label = ohe_label_df[0: int(ohe_speaker_context_party_df.shape[0]*0.8)]

test_data = ohe_speaker_context_party_df[int(ohe_speaker_context_party_df.shape[0]*0.8):int(ohe_speaker_context_party_df.shape[0])]
test_label = ohe_label_df[int(ohe_speaker_context_party_df.shape[0]*0.8):int(ohe_speaker_context_party_df.shape[0])]
# citation: https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.OneHotEncoder.html

# PyTorch imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim as optim
# citation: https://dev.to/vidyasagarmsc/pytorch-jupyter-notebook-modulenotfounderror-no-module-named-torch-2c1o
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
# citation: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

# build model
class speaker_context_party_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_sigmoid = nn.Sequential(
            nn.Linear(len(modified_data.columns), 1300),
            nn.Softmax(),
            nn.Linear(1300, 600),
            nn.Softmax(),
            nn.Linear(600, len(training_label.columns)),
            nn.Softmax()
        )

    def forward(self, x):
      output = self.linear_sigmoid(x)
      return output

speaker_context_party_model = speaker_context_party_nn().to(device)

x_data = torch.tensor(modified_data.to_numpy()).type(torch.float)
x_data = x_data.to(device)
y_label = torch.tensor(training_label.to_numpy()).type(torch.float)
y_label = y_label.to(device)

y_pred = speaker_context_party_model(x_data)
print(f"Predicted class: {y_pred}")

# citation: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
# citation: https://www.geeksforgeeks.org/converting-a-pandas-dataframe-to-a-pytorch-tensor/

# train model with for loop
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(speaker_context_party_model.parameters(), lr = 0.0001, momentum = 0.9)

epoch_num = 5
loss_list = []
train_loss = []
for epoch in range(epoch_num):
  speaker_context_party_model.train()
  for i in range(len(x_data)):
    x = x_data[i]
    prediction = speaker_context_party_model(x)
    loss_list.append(loss_func(prediction, y_label[i]))
    loss_func(prediction, y_label[i]).backward()
    optimizer.step()
# citation: https://www.youtube.com/watch?v=tHL5STNJKag

# save as pickle
# import pickle 
# with open('speaker_context_party_nn.pkl', 'wb') as f:  # open a text file
#     pickle.dump(speaker_context_party_model, f) # serialize the list
#     # f.close()

# save model
torch.save(speaker_context_party_model,"speaker_context_party_model_pytorch")