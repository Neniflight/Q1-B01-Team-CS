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
# import pickle
# import tensorflow
from tensorflow import keras
# import keras
from questions import predefined_questions
# import tensorflow_model_optimization as tfmot

load_dotenv()
genai.configure(api_key="AIzaSyCEdUpoG4Bs5QiZOcu2FZR7VNx719ZUifI")

import mesop as me
import mesop.labs as mel

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

@me.stateclass
class State:
  file: me.UploadedFile
  uploaded: bool = False
  overall_sens_score: float = 0.0
  overall_stance_score: float = 0.0
  overall_social_credibility: float = 0.0
  predefined_questions: list = field(default_factory=lambda: predefined_questions)
  chat_history: list[mel.ChatMessage]

def load(e: me.LoadEvent):
  me.set_theme_mode("system")

def handle_upload(event: me.UploadEvent):
  state = me.state(State)
  state.file = event.file
  state.uploaded = True

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
      # print(f"Question:{question}")
      response_generator = transform(question, state.chat_history)  
      response = ''.join(response_generator)
      # print(f"Response:{response}")
      time.sleep(5)   

def ask_pred_ai_social_cred(event: me.ClickEvent):
  state = me.state(State)
  # load model
  social_credit_model = keras.models.load_model("models/social_cred_predAI.h5")

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

  import pandas as pd
  import numpy as np
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
           label="Rate Article on Factuality Factors",
           on_click=ask_predefined_questions,
           color="primary",
           style = me.Style(border=me.Border.all(me.BorderSide(width=10, color="black")), align_self="center")
        )

        me.button(
          label="Rate Factuality Factors with PredAI",
          on_click=ask_pred_ai_social_cred,
          color="accent",
          style = me.Style(border=me.Border.all(me.BorderSide(width=10, color="black")), align_self="center")
      )

    with me.box(style=me.Style(height="50%")):
      mel.chat(
        transform, 
        title="Gemini Misinformation Helper", 
        bot_user="Chanly", # Short for the Vietnamese word for Truth
      )

    with me.box(style=me.Style(display='flex', width="100%", justify_content="space-around")):
      with me.box(style=me.Style(margin=me.Margin.all(15), border=me.Border.all(me.BorderSide(width=10, color="black")), border_radius=10, width="30%")):
        me.text(f"Overall Sensationalism: {state.overall_sens_score}", type="headline-5")
        me.progress_bar(mode="determinate", value=state.overall_sens_score*10, color='primary')
      with me.box(style=me.Style(margin=me.Margin.all(15), border=me.Border.all(me.BorderSide(width=10, color="black")), border_radius=10, width="30%")):
        me.text(f"Overall democratic Stance: {state.overall_stance_score}", type="headline-5")
        me.progress_bar(mode="determinate", value=state.overall_stance_score*10, color='primary')
      with me.box(style=me.Style(margin=me.Margin.all(15), border=me.Border.all(me.BorderSide(width=10, color="black")), border_radius=10, width="30%")):
        me.text(f"Overall Social credibility: {state.overall_social_credibility}", type="headline-5")
        me.progress_bar(mode="determinate", value=state.overall_social_credibility*10, color='primary')

def transform(input: str, history: list[mel.ChatMessage]):
    state = me.state(State)
    chat_history = ""
    if state.file and state.uploaded:
       chat_history += f"\nuser: {pdf_to_text(state.file)}"
    chat_history += "\n".join(f"{message.role}: {message.content}" for message in history)
    full_input = f"{chat_history}\nuser: {input}"
    time.sleep(5)
    # print(full_input)
    response = model.generate_content(full_input, stream=True)
    for chunk in response:
        text_chunk = chunk.text
        yield chunk.text
        overall_sens_match = re.search(r'overall\s*sensationalism\s*:\s*(\d+(\.\d+)?)', text_chunk, re.IGNORECASE)
        overall_stance_match = re.search(r'overall\s*stance\s*:\s*(\d+(\.\d+)?)', text_chunk, re.IGNORECASE)
        overall_social_credibility = re.search(r'overall\s*social\s*credibility\s*:\s*(\d+(\.\d+)?)', text_chunk, re.IGNORECASE)
        if overall_sens_match:
            state.overall_sens_score = float(overall_sens_match.group(1))
        if overall_stance_match:
            state.overall_stance_score = float(overall_stance_match.group(1))
        if overall_social_credibility:
            state.overall_social_credibility = float(overall_social_credibility.group(1))
    state.chat_history = history