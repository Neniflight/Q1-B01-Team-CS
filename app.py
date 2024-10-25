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

from questions import predefined_questions

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
  response_generator = transform("Give just the title of the article.", state.chat_history)
  response = ''.join(response_generator)
  print(response)
  time.sleep(3)

  subject_score = TextBlob(response).sentiment.subjectivity
  sentiment_analyzer = pipeline('sentiment-analysis', truncation=True)
  confidence = sentiment_analyzer(response)[0]['score']

  with open('data/speaker_reput_dict.pkl', 'rb') as file:
    speaker_reput_dict = pickle.load(file)
  
  speaker_generator = transform("Who is the speaker in this article? Only give the speaker name", state.chat_history)
  speaker = ''.join(speaker_generator)
  speaker = ('-'.join(speaker.split())).lower()
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
  state.veracity = round(np.mean([state.overall_naive_realism_score, 10 - state.overall_sens_score, state.overall_stance_score]), 2)


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
        me.text(f"Overall Democratic Stance: {state.overall_stance_score}", type="headline-5")
        me.progress_bar(mode="determinate", value=state.overall_stance_score*10, color='primary')
      with me.box(style=me.Style(margin=me.Margin.all(15), border=me.Border.all(me.BorderSide(width=10, color="black")), border_radius=10, width="30%")):
        me.text(f"Overall Naive Realism: {state.overall_naive_realism_score}", type="headline-5")
        me.progress_bar(mode="determinate", value=state.overall_naive_realism_score*10, color='primary')
    # veracity is calculated based on the mean of the factuality factors. If higher number in factuality factor correlates with more veracity, then we use that number. If the opposite behavior happens (aka if sensationalism is high in an article), we do (10 - factuality score)
    with me.box(style=me.Style(padding=me.Padding.all(15), margin=me.Margin.all(15), width="100%", align_items='center', justify_content='center', display='flex', flex_direction="column")):
      with me.box(style=me.Style(margin=me.Margin.all(15), border=me.Border.all(me.BorderSide(width=10, color="black")), border_radius=10, width="30%")):
        me.text(f"Veracity Score: {state.veracity}", type="headline-4", style=me.Style(font_weight="bold"))
        me.progress_bar(mode="determinate", value=(state.veracity)*10, color='primary')

def transform(input: str, history: list[mel.ChatMessage]):
    state = me.state(State)
    chat_history = ""
    if state.file and state.uploaded:
       chat_history += f"\nuser: {pdf_to_text(state.file)}"
    chat_history += "\n".join(f"{message.role}: {message.content}" for message in history)
    full_input = f"{chat_history}\nuser: {input}"
    time.sleep(1)
    # print(full_input)
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
    state.veracity = round(np.mean([state.overall_naive_realism_score, 10 - state.overall_sens_score, state.overall_stance_score]), 2)

