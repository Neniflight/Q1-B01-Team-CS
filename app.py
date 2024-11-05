import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
from dotenv import load_dotenv
import re
import mesop as me
import mesop.labs as mel
from dataclasses import field
import time
from PyPDF2 import PdfReader
import pickle
from transformers import pipeline
from textblob import TextBlob
import pandas as pd
import xgboost as xgb
import numpy as np
import chromadb
from tensorflow import keras
from questions import predefined_questions
# import tensorflow_model_optimization as tfmot

load_dotenv()
genai.configure(api_key="AIzaSyCEdUpoG4Bs5QiZOcu2FZR7VNx719ZUifI")

generation_config = {
    "max_output_tokens": 4000, # less output, means faster
    "response_mime_type": "text/plain",
    "temperature": 1, # higher temp --> more risks the model takes with choices
    "top_p": 0.95, # how many tokens are considered when producing outputs
    "top_k": 40, # token is selected from 40 likely tokens
}

safety_settings = {
  HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
  HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
  HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
  HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-002",
    generation_config=generation_config,
    safety_settings=safety_settings,
    system_instruction="You are trying to fight against misinformation by scoring different articles on their factuality factors. In your responses, do not use copyrighted material and be concise. Do not assess an article until you are given a factuality factor to grade on.",
)

chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_collection("Misinformation")

@me.stateclass
class State:
  file: me.UploadedFile
  uploaded: bool = False
  article_title: str = ""
  overall_sens_score: float = 0.0
  overall_stance_score: float = 0.0
  overall_social_credibility: float = 0.0
  predefined_questions: list = field(default_factory=lambda: predefined_questions)
  chat_history: list[mel.ChatMessage]
  overall_naive_realism_score: float = 0.0
  veracity: float = 0.0

def load(e: me.LoadEvent):
  me.set_theme_mode("system")

def handle_upload(event: me.UploadEvent):
  state = me.state(State)
  state.file = event.file
  state.uploaded = True
  get_headline(state.chat_history)

def pdf_to_text(user_pdf):
    reader = PdfReader(user_pdf)
    text = ""
    for i in range(len(reader.pages)):
      text_page = reader.pages[i]
      text += text_page.extract_text()
    return text
# citation: https://pypdf2.readthedocs.io/en/3.x/user/extract-text.html
# citation: https://www.geeksforgeeks.org/convert-pdf-to-txt-file-using-python/

def ask_predefined_questions(event: me.ClickEvent):
  state = me.state(State)
  for question in state.predefined_questions:
    print(f"Question:{question}")
    response_generator = transform(question, state.chat_history)  
    response = ''.join(response_generator)
    print(f"Response:{response}")
    time.sleep(5) 
    
def ask_pred_ai(event: me.ClickEvent):
  state = me.state(State)
  loaded_model = pickle.load(open("model/XGModel.sav", 'rb'))
  response = state.article_title
  print(response)

  subject_score = TextBlob(response).sentiment.subjectivity
  sentiment_analyzer = pipeline('sentiment-analysis', truncation=True)
  confidence = sentiment_analyzer(response)[0]['score']

  with open('data/speaker_reput_dict.pkl', 'rb') as file:
    speaker_reput_dict = pickle.load(file)
  
  speaker_generator = transform("Who is the speaker in this article? Only give the speaker name", state.chat_history)
  speaker = ''.join(speaker_generator)
  speaker = ('-'.join(speaker.split())).lower()
  speaker = speaker.replace("\n", "").replace(" ", "")
  #check if speaker has new line characters
  speaker_reput = speaker_reput_dict.get(speaker)
  print(speaker)

  df = pd.DataFrame({
     'barely_true_ratio': None if speaker_reput is None else speaker_reput[0],
     'false_ratio': None if speaker_reput is None else speaker_reput[1],
     'half_true_ratio': None if speaker_reput is None else speaker_reput[2],
     'mostly_true_ratio': None if speaker_reput is None else speaker_reput[3],
     'pants_on_fire_ratio': None if speaker_reput is None else speaker_reput[4],
     'confidence': confidence,
     'subjectivity': subject_score,
    }, index=[0])
  # {'barely-true': 0, 'false': 1, 'half-true': 2, 'mostly-true': 3, 'pants-fire': 4, 'true': 5}
  prediction = loaded_model.predict(df.loc[0:0])[0]
  prediction_to_score = {5: 10, 4: 0, 0: 4, 1: 2, 2: 6, 3: 8}
  state.overall_naive_realism_score = prediction_to_score[prediction]
  
  # load model
  social_credit_model = keras.models.load_model("model/social_cred_predAI.h5")

  # get info needed to input into model
  speaker_generator = transform("Give just the author of the article.", state.chat_history)
  time.sleep(5)
  speaker_response = ''.join(speaker_generator)
  print("speaker: " + speaker_response)
  context_generator = transform("Give just what type of text is this article without extra text or special characters.", state.chat_history)
  context_response = ''.join(context_generator)
  print("context: " + context_response)
  time.sleep(5)
  party_affli_generator = transform("Give only the party affiliation of the article without extra text or special characters.", state.chat_history)
  party_affli_response = ''.join(party_affli_generator)
  print("party_affli: " + party_affli_response)
  time.sleep(5)

  # Please ignore, this is for chekcing purposes and also for not using up API resource when I need to check features.
  # speaker_response = "scott-surovell"  
  # context_response = "a floor speech"
  # party_affli_response = "democrat"

  train_data = pd.read_csv('https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/refs/heads/master/dataset/tsv/train2.tsv', sep = "\t")
  first_data = train_data.columns
  train_data.loc[train_data.shape[0]] = first_data
  train_data.columns =['index','ID of statement', 'label', 'statement', 'subject', 'speaker', "speaker's job title", 'state info',
                     'party affiliation', 'barely true counts', 'false counts', 'half true counts', 'mostly true counts',
                    'pants on fire counts', 'context', 'extracted justification']
  train_data = train_data.drop(columns=['index'])
  train_data['subject'] = train_data['subject'].str.split(",")
  train_data["speaker's job title"] = train_data["speaker's job title"].str.lower()
  train_data["extracted justification"] = train_data["extracted justification"].str.split(" ")
  train_data = train_data.dropna()
  train_data = train_data[["label", "speaker", "context", "party affiliation"]]

  # extra step of cleaning for avg score based on labels
  modified_label = train_data.copy()
  modified_label['mod_label'] = modified_label['label'].replace({'pants-fire': 0, np.nan : 0, 'barely-true':2, 'false':4, 'half-true':6, 'mostly-true':8, 'true':10})
  modified_label = modified_label.drop(columns = ['label'])


  # EDA imported Speaker
  speaker_response = str.lower(speaker_response.replace(" ","-"))
  print(speaker_response)
  if speaker_response not in np.array(train_data["speaker"].unique()):
    print("oh no, assign speaker score to avg score")
    speaker_score = modified_label['mod_label'].mean()
  else:
    speaker_array = np.array(train_data['speaker'].unique())
    speaker_ohe = np.where(speaker_array == speaker_response, 1, 0)
    speaker_ohe = list(speaker_ohe.reshape(-1,1))

  # EDA imported context
  context_response = str.lower(context_response)
  print(context_response)
  if context_response not in np.array(train_data["context"].unique()):
    print("oh no, assign context score to avg score")
    context_score = modified_label['mod_label'].mean()
  else:
    context_array = np.array(train_data['context'].unique())
    context_ohe = np.where(context_array == context_response, 1, 0)
    context_ohe = list(context_ohe.reshape(-1,1))

  # EDA imported Party Affiliation
  party_affli_response = str.lower(party_affli_response.replace(" ","-"))
  print(party_affli_response)
  if party_affli_response not in np.array(train_data["party affiliation"].unique()):
    print("oh no, assign party score to avg score")
    party_affli_score = modified_label['mod_label'].mean()
  else:
    party_affli_array = np.array(train_data['party affiliation'].unique())
    party_affli_ohe = np.where(party_affli_array == party_affli_response, 1, 0)
    party_affli_ohe = list(party_affli_ohe.reshape(-1,1))

  # implement model prediction
  if speaker_response in np.array(train_data["speaker"]) and context_response in np.array(train_data["context"]) and party_affli_response in np.array(train_data["party affiliation"]):
    # convert ohe to df for input
    input = speaker_ohe + context_ohe + party_affli_ohe
    print(len(input))
    input_df = pd.DataFrame(input).T
    prediction = social_credit_model.predict(input_df)
    prediciton_list = prediction[0]
    for i in range(len(prediciton_list)):
       if prediciton_list[i] == max(prediciton_list):
          state.overall_social_credibility = i*2

  else:
    state.overall_social_credibility = (speaker_score + context_score + party_affli_score) / 3

  state.overall_social_credibility = round(state.overall_social_credibility, 2)
  
  print(state.overall_social_credibility)
  state.veracity = round(np.mean([state.overall_naive_realism_score, 10 - state.overall_sens_score, state.overall_stance_score, state.overall_social_credibility]), 2)

@me.page(path="/", title="Gemini Misinformation ChatBot")
def page():
    state = me.state(State)
    with me.box(style=me.Style(padding=me.Padding.all(15), margin=me.Margin.all(15), width="100%", align_items='center', justify_content='center', display='flex', flex_direction="column")):
      me.uploader(
        label="Upload PDF",
        accepted_file_types=[".pdf"],
        on_upload=handle_upload,
        type="flat",
        color="primary",
        style=me.Style(font_weight="bold"),
      )
      if state.uploaded:
        me.text("File uploaded!")
      me.button(
          label="Rate Factuality Factors with GenAI",
          on_click=ask_predefined_questions,
          color="primary",
          style = me.Style(border=me.Border.all(me.BorderSide(width=10, color="black")), align_self="center")
      )
      me.button(
          label="Rate Factuality Factors with PredAI",
          on_click=ask_pred_ai,
          color="accent",
          style = me.Style(border=me.Border.all(me.BorderSide(width=10, color="black")), align_self="center")
      )

    with me.box(style=me.Style(height="50%")):
      mel.chat(
        transform, 
        title="Gemini Misinformation Helper", 
        bot_user="Chanly", # Short for the Vietnamese word for Truth
      )

    with me.box(style=me.Style(display='flex', width="100%", justify_content="space-around", flex_wrap="wrap")):
      with me.box(style=me.Style(margin=me.Margin.all(15), border=me.Border.all(me.BorderSide(width=10, color="black")), border_radius=10, width="30%")):
        me.text(f"Overall Sensationalism: {state.overall_sens_score}", type="headline-5")
        me.progress_bar(mode="determinate", value=state.overall_sens_score*10, color='primary')
      with me.box(style=me.Style(margin=me.Margin.all(15), border=me.Border.all(me.BorderSide(width=10, color="black")), border_radius=10, width="30%")):
        me.text(f"Overall democratic Stance: {state.overall_stance_score}", type="headline-5")
        me.progress_bar(mode="determinate", value=state.overall_stance_score*10, color='primary')
      with me.box(style=me.Style(margin=me.Margin.all(15), border=me.Border.all(me.BorderSide(width=10, color="black")), border_radius=10, width="30%")):
        me.text(f"Overall Naive Realism: {state.overall_naive_realism_score}", type="headline-5")
        me.progress_bar(mode="determinate", value=state.overall_naive_realism_score*10, color='primary')
      with me.box(style=me.Style(margin=me.Margin.all(15), border=me.Border.all(me.BorderSide(width=10, color="black")), border_radius=10, width="30%")):
        me.text(f"Overall Social credibility: {state.overall_social_credibility}", type="headline-5")
        me.progress_bar(mode="determinate", value=state.overall_social_credibility*10, color='primary')
    # veracity is calculated based on the mean of the factuality factors. If higher number in factuality factor correlates with more veracity, then we use that number. If the opposite behavior happens (aka if sensationalism is high in an article), we do (10 - factuality score)
    with me.box(style=me.Style(padding=me.Padding.all(15), margin=me.Margin.all(15), width="100%", align_items='center', justify_content='center', display='flex', flex_direction="column")):
      with me.box(style=me.Style(margin=me.Margin.all(15), border=me.Border.all(me.BorderSide(width=10, color="black")), border_radius=10, width="30%")):
        me.text(f"Veracity Score: {state.veracity}", type="headline-4", style=me.Style(font_weight="bold"))
        me.progress_bar(mode="determinate", value=(state.veracity)*10, color='primary')

def get_headline(history: list[mel.ChatMessage]):
  state= me.state(State)
  chat_history = ""
  if state.file and state.uploaded:
    chat_history += f"\nuser: {pdf_to_text(state.file)}"
  chat_history += "\n".join(f"{message.role}: {message.content}" for message in history)
  full_input = f"{chat_history}\nuser: Give just the title of the article."
  time.sleep(2)
  response = model.generate_content(full_input, stream=True)
  full_response = "".join(chunk.text for chunk in response)
  state.article_title= full_response
  state.chat_history = history
  
def transform(input: str, history: list[mel.ChatMessage]):
    state = me.state(State)
    chat_history = ""
    if state.file and state.uploaded:
       chat_history += f"\nuser: {pdf_to_text(state.file)}"
    chat_history += "\n".join(f"{message.role}: {message.content}" for message in history)
    results = collection.query(query_texts=[state.article_title],
                                     n_results=3,
                                     where=
                                     {
                                        "label": "true"
                                     })
    chromadb_info = "\n".join(results['documents'][0])

    full_input = f"{chat_history}\nChromaDB Info: Based on the headline, these are the most similar true statements: {chromadb_info}\nuser: {input}"
    time.sleep(4)
    response = model.generate_content(full_input, stream=True)
    for chunk in response:
        text_chunk = chunk.text
        yield chunk.text
        overall_sens_match = re.search(r'overall\s*sensationalism\s*:\s*(\d+(\.\d+)?)', text_chunk, re.IGNORECASE)
        overall_stance_match = re.search(r'overall\s*stance\s*:\s*(\d+(\.\d+)?)', text_chunk, re.IGNORECASE)
        if overall_sens_match:
            state.overall_sens_score = float(overall_sens_match.group(1))
        if overall_stance_match:
            state.overall_stance_score = float(overall_stance_match.group(1))
    state.chat_history = history
    # FIX bug where if model is asked questions. veracity will automatically populate
    state.veracity = round(np.mean([state.overall_naive_realism_score, 10 - state.overall_sens_score, state.overall_stance_score, state.overall_social_credibility]), 2)
