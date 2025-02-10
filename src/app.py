
# Import necessary packages
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, content_types
from collections.abc import Iterable
import os
from dotenv import load_dotenv
from functools import partial
import re
import mesop as me
import mesop.labs as mel
from dataclasses import field
import time
from PyPDF2 import PdfReader
import pickle
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim as optim
from social_credibility_predAI_pytorch import speaker_context_party_nn
from transformers import pipeline
from textblob import TextBlob
import pandas as pd
import xgboost as xgb
import numpy as np
import chromadb
from questions import predefined_questions
from normal_prompting import normal_prompting_question
from fcot_prompting import fcot_prompting_question
from naive_realism import naive_realism_normal, naive_realism_cot, naive_realism_fcot

# from the env file, get your API_KEY
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

generation_config = {
    "max_output_tokens": 4000, # less output, means faster
    "response_mime_type": "text/plain",
    "temperature": 1, # higher temp --> more risks the model takes with choices
    "top_p": 0.95, # how many tokens are considered when producing outputs
    "top_k": 40, # token is selected from 40 likely tokens
}

# Turning all safety settings off
safety_settings = {
  HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
  HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
  HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
  HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

# function used to help with function calling and setting configurations
def tool_config_from_mode(mode: str, fns: Iterable[str] = ()):
    """Create a tool config with the specified function calling mode."""
    return content_types.to_tool_config(
        {"function_calling_config": {"mode": mode, "allowed_function_names": fns}}
    )

# Creating the model with proper configs
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-002",
    generation_config=generation_config,
    safety_settings=safety_settings,
    system_instruction="You are trying to fight against misinformation by scoring different articles on their factuality factors. In your responses, do not use copyrighted material and be concise. Do not assess an article until you are given a factuality factor to grade on.",
)

# Connecting to the vector database to allow for more informed prompts
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_collection("Misinformation")

@me.stateclass
class State:
  # Sets the initial states of different variables from microfactors to chat_history
  file: me.UploadedFile
  uploaded: bool = False
  article_title: str = ""
  overall_sens_score: float = 0.0
  overall_stance_score: float = 0.0
  overall_social_credibility: float = 0.0
  predefined_questions: list = field(default_factory=lambda: predefined_questions)
  normal_prompting_question: list = field(default_factory=lambda: normal_prompting_question)
  fcot_prompting_question: list = field(default_factory=lambda: fcot_prompting_question)
  chat_history: list[mel.ChatMessage]
  overall_naive_realism_score: float = 0.0
  veracity: float = 0.0
  overall_sens_normal_score: float = 0.0
  overall_stance_normal_score: float = 0.0
  overall_sens_fcot_score: float = 0.0
  overall_stance_fcot_score: float = 0.0
  normal_prompt_vs_fcot_prompt_log: dict[str, str] = field(default_factory=dict)
  vdb_response: str = ""
  serp_response: str = ""
  test_response: str = ""
  response: str = ""
  # citation for using dict: https://github.com/google/mesop/issues/814

# def page_load(e: me.LoadEvent):
#   state = me.state(State)
#   state.test_response = ""
#   state.vdb_response = ""
#   state.serp_response = ""

# Function used to handle when a user uploads a pdf file
def handle_upload(event: me.UploadEvent):
  state = me.state(State)
  state.file = event.file
  state.uploaded = True
  print('before get_headline')
  get_headline(state.chat_history)
  print('after get_headline')

def pdf_to_text(user_pdf):
  """converts the pdf the user uploads via the upload pdf button to string page by page
     and returns the text as a string.
    
    Args:
        user_pdf: A pdf the user uploads. 
  """
  reader = PdfReader(user_pdf)
  text = ""
  for i in range(len(reader.pages)):
    text_page = reader.pages[i]
    text += text_page.extract_text()
  return text
# citation: https://pypdf2.readthedocs.io/en/3.x/user/extract-text.html
# citation: https://www.geeksforgeeks.org/convert-pdf-to-txt-file-using-python/

from serpapi import GoogleSearch
from dotenv import load_dotenv
import os
from newspaper import Article
from newspaper import ArticleException
from datetime import datetime, timedelta
# serp_api_function
def serp_api(user_article_title):
  serp_api_key = os.getenv("SERP_API_KEY")
  # set up parameters for search
  params = {
  "engine": "google",
  "q": f"related: {user_article_title}",
#   "location": "Seattle-Tacoma, WA, Washington, United States", don't need location
  "hl": "en",
  "gl": "us",
  "google_domain": "google.com",
  "num": "10",
#   "start": "10",
  "safe": "active",
  "api_key": serp_api_key,
  "device": "desktop",
  }

  # Begin search and get organic results
  search = GoogleSearch(params)
  results = search.get_dict()
  organic_results = results["organic_results"]

  # create a function to find how many days, hours, or minutes ago the article was published
  def relative_date_to_absolute(relative_date):
    now = datetime.now()

    if "day" in relative_date:
        days = int(relative_date.split()[0])
        return (now - timedelta(days=days)).strftime('%Y-%m-%d')
    elif "hour" in relative_date:
        hours = int(relative_date.split()[0])
        return (now - timedelta(hours=hours)).strftime('%Y-%m-%d')
    elif "minute" in relative_date:
        minutes = int(relative_date.split()[0])
        return (now - timedelta(minutes=minutes)).strftime('%Y-%m-%d')
    else:
        # encountered bug here... relative_datetime is string.
        place_holder = now
        return datetime.strftime(place_holder, "%Y-%m-%d")
  
  # create a function to process the organic_results as dictionaries in a list
  def process_organic_results(results):
    similar_article_info = []
    irrelevant_texts = [
            "You have permission to edit this article.\n\nEdit Close",
            "Some other irrelevant text"
        ]
    for result in results:
        article_dict = {}
        try:
            link = result['link']
            article = Article(link, language='en')
            article.download()
            article.parse()
            article.nlp()
            article_dict['title'] = article.title 
            article_dict['authors'] = article.authors
            if article.text in irrelevant_texts:
                article_dict['summary'] = ''
                # article_dict['full_text'] = ''
            else:
                article_dict['summary'] = article.summary 
                # article_dict['full_text'] = article.text
                
            if article.publish_date:
                article_dict['publish_date'] = str(article.publish_date.date())
            else:
                article_dict['publish_date'] = relative_date_to_absolute(result.get('date'))
            article_dict['source'] = result['source']
            similar_article_info.append(article_dict)
        except (ArticleException, TypeError):
            article_dict['title'] = result['title']
            article_dict['authors'] = None
            article_dict['summary'] = result['snippet']
            # article_dict['full_text'] = None
            if result.get('date'):
                article_dict['publish_date'] = relative_date_to_absolute(result.get('date'))
            else:
                article_dict['publish_date'] = None
            article_dict['source'] = result['source']
            similar_article_info.append(article_dict)
    return similar_article_info
  
  # get similar articles using process_organic_results function
  similar_article_info = process_organic_results(organic_results)

  return similar_article_info

def ask_predefined_questions(event: me.ClickEvent):
  """loop through our predefined questions to ask gemini to give us a score of 1 to 10 
    for the sensationalism and political stance
    
    Args:
        event: this question is activated when the button associated with this function is clicked 
  """
  state = me.state(State)
  for question in state.predefined_questions:
    # print(f"Question:{question}")
    response_generator = transform(question, state.chat_history)  
    response = ''.join(response_generator)
    # print(f"Response:{response}")
    time.sleep(5)

def ask_normal_prompting_questions(event: me.ClickEvent):
  """loop through our normal prompted questions to ask gemini to give us a score of 1 to 10 
    for the sensationalism and political stance
    
    Args:
        event: this question is activated when the button associated with this function is clicked 
  """
  state = me.state(State)
  state.response = ''
  for question in state.normal_prompting_question:
    print(f"Question:{question}")
    response_generator = transform(question, state.chat_history)  
    state.response = ''.join(response_generator)
    print(f"Response:{state.response}")
    time.sleep(5)

def ask_fcot_prompting_questions(event: me.ClickEvent):
  """loop through our fractal chain of thought prompted questions (3 iterations) to ask gemini to give us a score of 1 to 10 
    for the sensationalism and political stance
    
    Args:
        event: this question is activated when the button associated with this function is clicked 
  """
  state = me.state(State)
  for question in state.fcot_prompting_question:
    # editing the question that will be going into gemini
    user_article_title = state.article_title
    articles_from_serp_api = serp_api(user_article_title)
    text_to_add = " Please also consider these articles' information in your analysis of the score." + str(articles_from_serp_api)
    question = question + text_to_add
    print("added serp_api info to fcot question")
    print("start asking fcot prompting questions")
    print(f"Question:{question}")
    response_generator = transform(question, state.chat_history)  
    response = ''.join(response_generator)
    print(f"Response:{response}")
    time.sleep(5)

def ask_prompting_questions_v2(vb: bool, serp: bool, fc: bool, prompt: str, event: me.ClickEvent):
  state = me.state(State)
  # reset the chat_history to nothing to avoid conflicts
  state.chat_history = []
  state.vdb_response = ""
  state.serp_response = ""
  input = ""
  if vb == True:
    user_article_title = state.article_title
    print(user_article_title)
    vdb_results = collection.query(query_texts=[user_article_title], n_results=3)
    vdb_results = str(vdb_results['metadatas'])
    vdb_str = f"ChromaDB Info: Based on the headline, these are the most similar statements: {vdb_results}"
    state.vdb_response = vdb_str
    input = input + vdb_str + "\n"
  if serp == True:
    user_article_title = state.article_title
    articles_from_serp_api = str(serp_api(user_article_title))
    print(user_article_title)
    serp_str = f"SERP API: These are similar articles found online via an API. Please consider these articles' information in the score: {articles_from_serp_api}"
    state.serp_response = serp_str
    input = input + serp_str + "\n"
  input = input + prompt
  response_generator = transform_test(input, state.chat_history)
  response = ''.join(response_generator)
  state.test_response = response
  print(f"Response:{response}")
  time.sleep(5)

    
def ask_pred_ai(event: me.ClickEvent):
  """Runs two Predictive AI models (sentiment_analyzer & social_credit_model) and returns a score for each factuality factors
    
    Args:
        event: this question is activated when the button associated with this function is clicked 
  """
  state = me.state(State)
  loaded_model = pickle.load(open("../model/XGModel.sav", 'rb'))
  response = state.article_title
  print(response)

  subject_score = TextBlob(response).sentiment.subjectivity
  sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', truncation=True)
  confidence = sentiment_analyzer(response)[0]['score']

  with open('../data/speaker_reput_dict.pkl', 'rb') as file:
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
  social_credit_model = speaker_context_party_nn()
  state_dict = torch.load("../model/speaker_context_party_model_state.pth")
  social_credit_model.load_state_dict(state_dict)
  print("hello loaded model!")
  # citation: https://discuss.pytorch.org/t/error-loading-saved-model/8371/6


  # Please ignore, this is for chekcing purposes and also for not using up API resource when I need to check features.
  # speaker_response = "scott-surovell"  
  # context_response = "a floor speech"
  # party_affli_response = "democrat"

  train_data = pd.read_csv('https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/refs/heads/master/dataset/tsv/train2.tsv', sep = "\t")
  train_data.columns =['index','ID of statement', 'label', 'statement', 'subject', 'speaker', "speaker's job title", 'state info',
                     'party affiliation', 'barely true counts', 'false counts', 'half true counts', 'mostly true counts',
                    'pants on fire counts', 'context', 'extracted justification']
  train_data = train_data[["label", "speaker", "context", "party affiliation"]]
  train_data = train_data.dropna()

  # extra step of cleaning for avg score based on labels
  modified_label = train_data.copy()
  modified_label['mod_label'] = modified_label['label'].replace({'pants-fire': 0, np.nan : 0, 'barely-true':2, 'false':4, 'half-true':6, 'mostly-true':8, 'true':10})
  modified_label.head()

  # clean up context
  # write function to clean context
  import nltk
  from nltk.corpus import stopwords
  nltk.download('stopwords')
  unique_characters = '!@#$%^&*()_+~{}|:"<>?,./;'

  # light cleaning (fixing syntax)
  def clean_context(context):
    """cleaned the context column of the liar plus dataset by deleting stopwords
    
    Args:
        context: each datapoint in the context column from the liar plus dataset 
    """
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
    """going through the context column of the liar plus dataset and categorizing them into
       more generalized categories based on what text was included in the original datapoint
       after deleting stop words 
    
    Args:
        context: each datapoint in the context column from the liar plus dataset 
    """
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

  # drop the empty outputs, weird/unique context
  modified_label = modified_label[modified_label['context'] != ""]

  unique_contexts = list(modified_label.context.unique())
  unique_party = list(modified_label['party affiliation'].unique())
  # citation: https://www.geeksforgeeks.org/removing-stop-words-nltk-python/

# get info needed to input into model
  speaker_response = speaker
  context_generator = transform("What type of text is this article without extra text or special characters chosing one from the list: " + str(unique_contexts), state.chat_history)
  context_response = ''.join(context_generator)
  context_response = context_response.replace("\n", "")
  print("context: " + context_response)
  time.sleep(5)
  party_affli_generator = transform("Give only the party affiliation of the article without extra text or special characters chosing one from the list: " + str(unique_party), state.chat_history)
  party_affli_response = ''.join(party_affli_generator)
  party_affli_response = party_affli_response.replace('\n', '')
  print("party_affli: " + party_affli_response)
  time.sleep(5)

  # Please ignore, this is for testing purposes.
#   speaker_response = "scott-surovell"  
#   context_response = "speech"
#   party_affli_response = "democrat"

  # EDA imported Speaker
  speaker_response = str.lower(speaker_response.replace(" ","-"))
  print(speaker_response)
  if speaker_response not in np.array(modified_label["speaker"].unique()):
    print("oh no, assign speaker score to avg score")
    speaker_score = modified_label['mod_label'].mean()
  else:
    speaker_array = np.array(modified_label['speaker'].unique())
    speaker_ohe = np.where(speaker_array == speaker_response, 1, 0)
    speaker_ohe = list(speaker_ohe.reshape(-1,1))
    print("speaker_ohe len: " + str(len(speaker_ohe)))

  # EDA imported context
  context_response = str.lower(context_response)
  print(context_response)
  if context_response not in np.array(modified_label["context"].unique()):
    print("oh no, assign context score to avg score")
    context_score = modified_label['mod_label'].mean()
  else:
    context_array = np.array(modified_label['context'].unique())
    context_ohe = np.where(context_array == context_response, 1, 0)
    context_ohe = list(context_ohe.reshape(-1,1))
    print("context_ohe len: " + str(len(context_ohe)))

  # EDA imported Party Affiliation
  party_affli_response = str.lower(party_affli_response.replace(" ","-"))
  print(party_affli_response)
  if party_affli_response not in np.array(modified_label["party affiliation"].unique()):
    print("oh no, assign party score to avg score")
    party_affli_score = modified_label['mod_label'].mean()
  else:
    party_affli_array = np.array(modified_label['party affiliation'].unique())
    party_affli_ohe = np.where(party_affli_array == party_affli_response, 1, 0)
    party_affli_ohe = list(party_affli_ohe.reshape(-1,1))
    print("party_affli_ohe len: " + str(len(party_affli_ohe)))

  # get device
  device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
  )

  # implement model prediction
  if speaker_response in np.array(modified_label["speaker"]) and context_response in np.array(modified_label["context"]) and party_affli_response in np.array(modified_label["party affiliation"]):
    # convert ohe to df for input
    input = speaker_ohe + context_ohe + party_affli_ohe
    print(len(input))
    input_df = pd.DataFrame(input).T
    # modify data to become torch.tensor
    input_x = torch.tensor(input_df.to_numpy()).type(torch.float)
    # input_x = input_x.to(device)
    print("input_x len : " + str(len(input_x[0])))
    prediction = social_credit_model(input_x[0])
    prediciton_list = prediction.tolist()
    for i in range(len(prediciton_list)):
       if prediciton_list[i] == max(prediciton_list):
          state.overall_social_credibility = i*2
    print(prediction)
    # state.overall_social_credibility = prediction
  else:
    state.overall_social_credibility = (speaker_score + context_score + party_affli_score) / 3
  print(f"This is overall_social_credibility: {state.overall_social_credibility}")
  state.overall_social_credibility = round(state.overall_social_credibility, 2)
  
  print(state.overall_social_credibility)
  state.veracity = round(np.mean([state.overall_naive_realism_score, 10 - state.overall_sens_score, state.overall_stance_score, state.overall_social_credibility]), 2)

def navigate_to(event: me.ClickEvent, path: str):
  me.navigate(path)

navigate_to_ve = partial(navigate_to, path="/Gemini_Misinformation_ChatBot")
navigate_to_normal = partial(navigate_to, path="/normal_adjustments")
navigate_to_cot = partial(navigate_to, path="/cot_adjustments")
navigate_to_fcot = partial(navigate_to, path="/fcot_adjustments")

def create_reproducible_page(header_text: str, placeholder_text: str, prompt_adjust: tuple):
  state = me.state(State)
  with me.box(style=me.Style(padding=me.Padding.all(15), margin=me.Margin.all(15), width="100%", align_items='center', justify_content='center', flex_direction="column")):
      me.uploader(
      label="Upload PDF",
      accepted_file_types=[".pdf"],
      on_upload=handle_upload,
      type="flat",
      color="primary",
      style=me.Style(font_weight="bold"),
      )
      if state.uploaded:
        me.text("File uploaded already!")
      me.text(
          header_text,
          style=me.Style(
              font_size=24,
              font_weight="bold",
              margin=me.Margin(bottom=20)
          )
      )
      me.text(
          "Prompt",
          style=me.Style(
              font_size=20,
              font_weight="bold",
              margin=me.Margin(bottom=20)
          )
      )
      me.textarea(
        value= placeholder_text,
        appearance="outline",
        readonly=True,
        style=me.Style(width="100%", padding=me.Padding.all(15))
      )
      me.button(
          "Run Prompt",
          on_click=lambda e: ask_prompting_questions_v2(prompt_adjust[0], prompt_adjust[1], prompt_adjust[2], placeholder_text, e),
          color="primary",
          type="flat",
          style=me.Style(
              align_self="center",
              border=me.Border.all(me.BorderSide(width=2, color="black")),
          )
      )
      me.text(
          "Response",
          style=me.Style(
              font_size=20,
              font_weight="bold",
              margin=me.Margin(bottom=20)
          )
      )
      me.textarea(
        value=state.test_response,
        readonly=True,
        style=me.Style(width="100%", padding=me.Padding.all(15))
      )
      me.text(
          "Vector Database",
          style=me.Style(
              font_size=20,
              font_weight="bold",
              margin=me.Margin(bottom=20)
          )
      )
      me.textarea(
        value=state.vdb_response,
        readonly=True,
        style=me.Style(width="100%", padding=me.Padding.all(15))
      )
      me.text(
          "SERP API",
          style=me.Style(
              font_size=20,
              font_weight="bold",
              margin=me.Margin(bottom=20)
          )
      )
      me.textarea(
        value=state.serp_response,
        readonly=True,
        style=me.Style(width="100%", padding=me.Padding.all(15))
      )

@me.page(path='/',stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap"
    ])
def home():
  # see whether it is good on my computer
  with me.box(style=me.Style(width="100%", height="100vh", background="white", flex_direction="column", justify_content="flex-start", align_items="center", display="flex", margin=me.Margin.all(0), overflow="auto")):
    with me.box(style=me.Style(position='fixed', width="100%", display='flex', top=0, overflow='hidden', justify_content="space-between", align_content="center", background='white', border=me.Border(bottom=me.BorderSide(width="0.5px", color='#010021', style='solid')), padding=me.Padding.symmetric(vertical=15, horizontal=50), z_index=10)):
      me.image(src="https://res.cloudinary.com/dd7kwlela/image/upload/v1738889378/capstone-dsc180b/jiz38dkxevducq0rpeye.png", style=me.Style(height=48))
      with me.box(style=me.Style(justify_content="flex-start", align_items="center", gap=40, display="flex")):
        me.link(text="Try Chenly Insights", url="/", style=me.Style(text_decoration='none', font_family='Inter', color="white", font_size=16, font_weight='bold', background="#010021", padding=me.Padding.symmetric(vertical=8, horizontal=10), border_radius=5))
        me.link(text="Prompt Testing", url="/", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
        me.link(text="Pipeline Explanation", url="/", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
        me.link(text="About Us", url="/", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
    with me.box(style=me.Style(max_width=1440, padding=me.Padding.all(100), justify_content="space-between", align_items="flex-start", display="inline-flex")):
      with me.box(style=me.Style(height="331px", flex_direction="column", justify_content= "flex-start", align_items="flex-start", display="inline-flex")):
        with me.box(style=me.Style(padding=me.Padding.symmetric(vertical=50), flex_direction="column", justify_content="center", align_items="flex-start", gap="10px")):
          me.html("""
            <div style="background: linear-gradient(to right, #5271FF, #22BB7C); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Inter', sans-serif; font-size: 70px; font-weight: 700; margin: 0;">
              Chenly Insights
            </div>
          """, mode='sandboxed', style=me.Style(width="100%", height=100, white_space="nowrap", margin=me.Margin.all(0)))
          # me.text(text="Chenly Insights", type="headline-1", style=me.Style(background="linear-gradient(to right, #5271FF, #22BB7C)", font_family="Inter",  font_weight=700, "-webkit-background-clip": "text", "-webkit-text-fill-color": "transparent"))
          me.text(text="Fight Against Misinformation", type="headline-3", style=me.Style(color="#010021", word_wrap="break-word",font_family="Inter", font_weight=700, margin=me.Margin.all(5)))
          me.text(text="By Calvin Nguyen and Samantha Lin", type="headline-5", style=me.Style(color="#A5A5A3", word_wrap="break-word", font_family="Inter", font_weight=700, margin=me.Margin.all(5)))
        with me.box(style=me.Style(justify_content="flex-start", align_items="center", gap=5, display="inline-flex")):
          me.button("Try It Out!", type='flat', style=me.Style(font_family="Inter", font_size="20px", font_weight="bold"))
          me.button("Learn More", style=me.Style(font_family="Inter", font_size="20px", font_weight="bold"))
      me.image(src="https://media.istockphoto.com/id/1409182397/vector/spreading-fake-news-concept.jpg?s=2048x2048&w=is&k=20&c=veFBTYmO-wHEh7khaQ8wQWJN0-DO4Q2hQY7lrboIGbg=", style=me.Style(width="540px", height="485px"))
    with me.box(style=me.Style(align_self="stretch", background="linear-gradient(90deg, #5271FF 0%, #22BB7C 100%)", justify_content="center", display='flex')):
      with me.box(style=me.Style(width="100%", padding=me.Padding.all(100), max_width=1440, flex_direction="column", justify_content="flex-start", align_items= "flex-start", gap=20, display="inline-flex")):
        me.text("Samples", type="headline-3", style=me.Style(align_self="stretch", color="white", font_family="Inter", font_weight=700, word_wrap="break-word", margin=me.Margin.all(0)))
        with me.box(style=me.Style(justify_content="center", align_items="flex-start", gap=30, display="inline-flex")):
          me.button("Trusted News Articles", type="stroked", style=me.Style(color="white", font_family="Inter", font_weight="bold", font_size="20px"))
          me.button("Satirical Articles", type="stroked", style=me.Style(color="white", font_family="Inter", font_weight="bold", font_size="20px"))
          me.button("Sketchy Sources", type="stroked", style=me.Style(color="white", font_family="Inter", font_weight="bold", font_size="20px"))
        with me.box(style=me.Style(display="flex", align_self="stretch", justify_content="center")):
          me.image(src="https://archive.org/download/placeholder-image/placeholder-image.jpg", style=me.Style(align_self="stretch", border_radius="10px", height="572px"))
    with me.box(style=me.Style(max_width=1440, padding=me.Padding.all(100), background="white", justify_content="space-between", align_items="flex-start", display="inline-flex")):
      with me.box(style=me.Style(flex="1 1 0", padding= me.Padding(right=50), flex_direction="column", justify_content="flex-start", align_items="flex_start", gap=15, display="flex")):
        me.text("Prompt Testing", type='headline-3', style=me.Style(color="#010021", font_family="Inter", font_weight="bold", word_wrap="break-word", margin=me.Margin.all(0)))
        me.markdown("""
Here, you can test out different prompting techniques along with any adjustments that could be made to them. You can upload or link your own article and test it on the criteria for the factuality factor of **Naive Realism**. In total, there are **24 different ways** you could prompt!  
There are **3 types** of prompting:  
- **Normal**  
- **Chain of Thought (COT)**  
- **Fractal Chain of Thought (FCOT)**  
                    
Additionally, these prompts can be accompanied with a combination of the following:   
- **Google Engine Search via SERP**  
- **Vector Database Lookup**  
- **Function Call**  
          """, style=me.Style(font_family="Inter", font_size="16px", color="#010021"))
        me.button("Try It Out", type='flat', style=me.Style(font_family="Inter", font_size="20px", font_weight="bold"))
      with me.box(style=me.Style(align_self="stretch", flex_direction="column", justify_content="flex-start", align_items="flex-start", gap="33px", display="inline-flex")):
        with me.box(style=me.Style(justify_content="flex-start", align_items="center", gap=75, display="inline-flex")):
          with me.box(style=me.Style(flex_direction="column", justify_content="center", align_items="center", gap=20, display="inline-flex")):
            me.image(src="https://res.cloudinary.com/dd7kwlela/image/upload/v1738885691/capstone-dsc180b/fvui2nlaxlrxilubioil.png", style=me.Style(width=94, height=94))
            me.text("Normal", type="headline-6", style=me.Style(font_family="Inter", font_weight='bold'))
          with me.box(style=me.Style(flex_direction="column", justify_content="center", align_items="center", gap=20, display="inline-flex")):
            me.image(src="https://res.cloudinary.com/dd7kwlela/image/upload/v1738885691/capstone-dsc180b/jnbbfo7a8o4n9yomjnpi.png", style=me.Style(width=94, height=94))
            me.text("CoT", type="headline-6", style=me.Style(font_family="Inter", font_weight='bold'))
          with me.box(style=me.Style(flex_direction="column", justify_content="center", align_items="center", gap=20, display="inline-flex")):
            me.image(src="https://res.cloudinary.com/dd7kwlela/image/upload/v1738885691/capstone-dsc180b/ejlgqhtuusoxuybs7jhz.png", style=me.Style(width=94, height=94))
            me.text("CoT", type="headline-6", style=me.Style(font_family="Inter", font_weight='bold'))
        me.image(src="https://res.cloudinary.com/dd7kwlela/image/upload/v1738886394/capstone-dsc180b/trntngqmwxwoxvcnbkw5.png", style=me.Style(align_self="stretch", width=450, border_radius=10, height="auto"))
    with me.box(style=me.Style(align_self="stretch", justify_content="center", display='flex', border=me.Border(top=me.BorderSide(width="1px", color="#A5A5A3", style="solid")))):
      with me.box(style=me.Style(max_width=1440, padding=me.Padding.all(100), background="white", flex_direction='column', justify_content='flex-start', align_items='flex-start', gap=50, display='inline-flex')):
        with me.box(style=me.Style(align_self='stretch', flex_direction="column", justify_content="center", align_items="flex-start", gap=10, display='flex')):
          me.text("About", type='headline-3', style=me.Style(font_family="Inter", font_weight='bold', color='#010021', margin=me.Margin.all(0)))
          with me.box(style=me.Style(align_self="stretch", justify_content='space-between', align_items='flex-start', display='inline-flex')):
            me.text("This project was done for DSC 180 Capstone for the Winter 2025 Showcase. We collaborated with Dr. Ali Arsanjani, a Director at Google AI, on the subject “GenAI for Good”, where we fight against misinformation present in our society with AI. As a group of two, we present the Chenly Insights. A cutting edge technology that uses the latest generative and predictive AI to fight against misinformation in news article. You can upload or link different news articles and received detail breakdowns of its truthfulness depending on different factuality factors. You can view more information about our tool and its paper through the “Learn More” button. ", type="body-1", style=me.Style(font_family="Inter", color="#010021", word_wrap="break-word"))
            me.image(src="https://res.cloudinary.com/dd7kwlela/image/upload/v1738889378/capstone-dsc180b/jiz38dkxevducq0rpeye.png", style=me.Style(margin=me.Margin(left=30), width=430, height='auto'))
        with me.box(style=me.Style(align_self="stretch", justify_content="center", align_items="flex-start", gap=150, display="inline-flex")):
          with me.card(appearance="outlined", style=me.Style(font_family="Inter")):
            me.html("""
            <div style="background: linear-gradient(to right, #5271FF, #22BB7C); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Inter', sans-serif; font-size: 32px; font-weight: 700; margin: 0;">
              Calvin Nguyen
            </div>
          """, mode='sandboxed', style=me.Style(width="60%", height=55, white_space="nowrap", margin=me.Margin.all(0)))
            me.html("""
            <div style="background: linear-gradient(to right, #5271FF, #22BB7C); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Inter', sans-serif; font-size: 20px; font-weight: 700; margin: 0;">
              Data Science Major
            </div>
          """, mode='sandboxed', style=me.Style(width="40%", height=40, white_space="nowrap", margin=me.Margin.all(0)))
            me.image(
              style=me.Style(
                width="100%",
              ),
              src="https://res.cloudinary.com/dd7kwlela/image/upload/v1738913112/capstone-dsc180b/sy6z0sv7rij8h8wfq3dg.png",
            )
            with me.card_content():
              me.text(
                "I'm a senior Data Science major with a minor in Design. I love gaming, playing music, and machine learning."
              )

            with me.card_actions(align="end"):
              me.button(label="Linkedin", type="flat", style=me.Style(font_family="Inter", margin=me.Margin.symmetric(horizontal=10), background="#5271FF", color="white"))
              me.button(label="Github", type="flat", style=me.Style(font_family="Inter", background="#010021", color="white"))
          
          with me.card(appearance="outlined", style=me.Style(font_family="Inter")):
            me.html("""
            <div style="background: linear-gradient(to right, #5271FF, #22BB7C); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Inter', sans-serif; font-size: 32px; font-weight: 700; margin: 0;">
              Samantha Lin
            </div>
          """, mode='sandboxed', style=me.Style(width="60%", height=55, white_space="nowrap", margin=me.Margin.all(0)))
            me.html("""
            <div style="background: linear-gradient(to right, #5271FF, #22BB7C); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Inter', sans-serif; font-size: 20px; font-weight: 700; margin: 0;">
              Data Science Major
            </div>
          """, mode='sandboxed', style=me.Style(width="45%", height=40, white_space="nowrap", margin=me.Margin.all(0)))
            me.image(
              style=me.Style(
                width="100%",
              ),
              src="https://res.cloudinary.com/dd7kwlela/image/upload/v1739064243/capstone-dsc180b/hvrc5w1len1npug2mk8o.png",
            )
            with me.card_content():
              me.text(
                "I’m a senior Data Science major with a minor in Business. I love watching anime and listening to music!"
              )

            with me.card_actions(align="end"):
              me.button(label="Linkedin", type="flat", style=me.Style(font_family="Inter", margin=me.Margin.symmetric(horizontal=10), background="#5271FF", color="white"))
              me.button(label="Github", type="flat", style=me.Style(font_family="Inter", background="#010021", color="white"))

@me.page(path="/normal_adjustments")
def normal_adjustments():
  with me.box(style=me.Style(padding=me.Padding.all(15), margin=me.Margin.all(15), width="100%", align_items='center', justify_content='center', flex_direction="column")):
    me.text("Normal Prompting Adjustments", type="headline-3", style=me.Style(margin=me.Margin(bottom=42)))
    me.text("Click on the buttons below to apply the selected adjustment", type="headline-5", style=me.Style(margin=me.Margin(bottom=42)))
    with me.box(style=me.Style(display="flex", flex_direction="row", gap=25)):
      me.button("None", on_click=lambda x: navigate_to(x, "/norm_none"), color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Vector Database", on_click=lambda x: navigate_to(x, "/norm_vb"), color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Serp API", on_click=lambda x: navigate_to(x, "/norm_serp"), color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Function Calling", on_click=navigate_to_ve, color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Vector Database & Serp API", on_click=lambda x: navigate_to(x, "/norm_vb_serp"), color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Vector Database & Function Calling", on_click=navigate_to_ve, color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Serp API and Function Calling", on_click=navigate_to_ve, color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Vector Database & Serp API & Function Calling", on_click=navigate_to_ve, color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))

@me.page(path="/norm_none")
def norm_none():
  create_reproducible_page("Normal Prompting with No Adjustments", naive_realism_normal[0], (False, False, False))

@me.page(path="/norm_vb")
def norm_vb():
  create_reproducible_page("Normal Prompting with Vector Database", naive_realism_normal[0], (True, False, False))

@me.page(path="/norm_serp")
def norm_serp():
  create_reproducible_page("Normal Prompting with SERP API", naive_realism_normal[0], (False, True, False))

@me.page(path="/norm_vb_serp")
def norm_serp():
  create_reproducible_page("Normal Prompting with Vector Database and SERP API", naive_realism_normal[0], (True, True, False))

@me.page(path="/cot_adjustments")
def cot_adjustments():
  with me.box(style=me.Style(padding=me.Padding.all(15), margin=me.Margin.all(15), width="100%", align_items='center', justify_content='center', flex_direction="column")):
    me.text("Chain of Thought Prompting Adjustments", type="headline-3", style=me.Style(margin=me.Margin(bottom=42)))
    me.text("Click on the buttons below to apply the selected adjustment", type="headline-5", style=me.Style(margin=me.Margin(bottom=42)))
    with me.box(style=me.Style(display="flex", flex_direction="row", gap=25)):
      me.button("None", on_click=lambda x: navigate_to(x, "/cot_normal"), color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Vector Database", on_click=lambda x: navigate_to(x, "/cot_vb"), color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Serp API", on_click=lambda x: navigate_to(x, "/cot_serp"), color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Function Calling", on_click=navigate_to_ve, color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Vector Database & Serp API", on_click=lambda x: navigate_to(x, "/cot_vb_serp"), color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Vector Database & Function Calling", on_click=navigate_to_ve, color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Serp API and Function Calling", on_click=navigate_to_ve, color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Vector Database & Serp API & Function Calling", on_click=navigate_to_ve, color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))

@me.page(path="/cot_normal")
def cot_none():
  create_reproducible_page("Chain of Thought Prompting with No Adjustments", naive_realism_cot[0], (False, False, False))

@me.page(path="/cot_vb")
def cot_vb():
  create_reproducible_page("Chain of Thought Prompting with Vector Database", naive_realism_cot[0], (True, False, False))

@me.page(path="/cot_serp")
def cot_serp():
  create_reproducible_page("Chain of Thought Prompting with SERP API", naive_realism_cot[0], (False, True, False))

@me.page(path="/cot_vb_serp")
def cot_serp():
  create_reproducible_page("Chain of Thought Prompting with Vector Database and SERP API", naive_realism_cot[0], (True, True, False))

@me.page(path="/fcot_adjustments")
def fcot_adjustments():
  with me.box(style=me.Style(padding=me.Padding.all(15), margin=me.Margin.all(15), width="100%", align_items='center', justify_content='center', flex_direction="column")):
    me.text("Fractal Chain of Thought Prompting Adjustments", type="headline-3", style=me.Style(margin=me.Margin(bottom=42)))
    me.text("Click on the buttons below to apply the selected adjustment", type="headline-5", style=me.Style(margin=me.Margin(bottom=42)))
    with me.box(style=me.Style(display="flex", flex_direction="row", gap=25)):
      me.button("None", on_click=lambda x: navigate_to(x, "/fcot_normal"), color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Vector Database", on_click=lambda x: navigate_to(x, "/fcot_vb"), color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Serp API", on_click=lambda x: navigate_to(x, "/fcot_serp"), color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Function Calling", on_click=navigate_to_ve, color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Vector Database & Serp API", on_click=lambda x: navigate_to(x, "/fcot_vb_serp"), color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Vector Database & Function Calling", on_click=navigate_to_ve, color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Serp API and Function Calling", on_click=navigate_to_ve, color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))
      me.button("Vector Database & Serp API & Function Calling", on_click=navigate_to_ve, color="primary", type="flat", style = me.Style(border=me.Border.all(me.BorderSide(width=2, color="black")), align_self="center"))

@me.page(path="/fcot_normal")
def fcot_none():
  create_reproducible_page("Fractal Chain of Thought Prompting with No Adjustments", naive_realism_fcot[0], (False, False, False))

@me.page(path="/fcot_vb")
def fcot_vb():
  create_reproducible_page("Fractal Chain of Thought Prompting with Vector Database", naive_realism_fcot[0], (True, False, False))

@me.page(path="/fcot_serp")
def fcot_serp():
  create_reproducible_page("Fractal Chain of Thought Prompting with SERP API", naive_realism_fcot[0], (False, True, False))

@me.page(path="/fcot_vb_serp")
def fcot_serp():
  create_reproducible_page("Fractal Chain of Thought Prompting with Vector Database and SERP API", naive_realism_fcot[0], (True, True, False))


@me.page(path="/Gemini_Misinformation_ChatBot")
def Gemini_Misinformation_ChatBot():
    """
    setting up the mesop interface including the chatbox, buttons, and scores bars for each factuality factors
    """
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
          label="Ask Normal Prompt Questions",
          on_click=ask_normal_prompting_questions,
          color="primary",
          style = me.Style(border=me.Border.all(me.BorderSide(width=10, color="black")), align_self="center")
      )
      me.button(
          label="Ask fcot Prompt Questions",
          on_click=ask_fcot_prompting_questions,
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

    # Contains factuality factor and veracity scores
    with me.box(style=me.Style(display='flex', width="100%", justify_content="space-around", flex_wrap="wrap")):
      with me.box(style=me.Style(margin=me.Margin.all(15), border=me.Border.all(me.BorderSide(width=10, color="black")), border_radius=10, width="30%")):
        me.text(f"Overall Sensationalism: {state.overall_sens_score}", type="headline-5")
        me.progress_bar(mode="determinate", value=state.overall_sens_score*10, color='primary')
      with me.box(style=me.Style(margin=me.Margin.all(15), border=me.Border.all(me.BorderSide(width=10, color="black")), border_radius=10, width="30%")):
        me.text(f"Overall Political Stance: {state.overall_stance_score}", type="headline-5")
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

    # create dataframe for table view in mesop interface
    # using the response output for everything in table for now, will switch to the best prompting technique after
    # we find out the best combo
    overall_naive_realism_normal_score = re.search(r'overall\s*naive\s*realism\s*:\s*(\d+(\.\d+)?)', state.test_response, re.IGNORECASE)
    overall_sens_normal_score = re.search(r'normal\s*prompting\s*overall\s*sensationalism\s*:\s*(\d+(\.\d+)?)', state.response , re.IGNORECASE)
    overall_stance_normal_score = re.search(r'normal\s*prompting\s*overall\s*stance\s*:\s*(\d+(\.\d+)?)', state.response, re.IGNORECASE)
    score_table = pd.DataFrame(
      data = {
        "FACTUALITY FACTOR": ["Sensationalism", "Political Stance", "Naive Realism", "Social Credibility"],
        "SCORE": [overall_sens_normal_score, overall_stance_normal_score, overall_naive_realism_normal_score, state.overall_social_credibility],
        "CONSIDERATION": [state.response, state.response, state.test_response, "N/A calculated by Predictive AI"],
        "CITATION": ["gemini-1.5-pro-002",'gemini-1.5-pro-002','gemini-1.5-pro-002', 'Liar Plus dataset, Pytorch Neural Network']
      }
    )

    with me.box(style=me.Style(padding=me.Padding.all(15), margin=me.Margin.all(15))):
      me.table(score_table)

# not sure why its giving me an error commenting it out for now
# @me.page(path="/combined", title="Pred and Generative")
# def page():
#   state = me.state(State)
#   with me.box(style=me.Style(padding=me.Padding.all(15), margin=me.Margin.all(15), width="100%", align_items='center', justify_content='center', display='flex', flex_direction="column")):
#     me.text("Generative AI with function calling", type='headline-3')
#     me.uploader(
#       label="Upload PDF",
#       accepted_file_types=[".pdf"],
#       on_upload=handle_upload,
#       type="flat",
#       color="primary",
#       style=me.Style(font_weight="bold"),
#     )
#     if state.uploaded:
#       me.text("File uploaded!")
#   with me.box(style=me.Style(height="50%")):
#     mel.chat(
#       transform_fc, 
#       title="Gemini Misinformation Helper", 
#       bot_user="Chanly", # Short for the Vietnamese word for Truth
#     )

# Used to get the headline of an article 
def get_headline(history: list[mel.ChatMessage]):
  """asks gemini to go through the text of the pdf uploaded by the user and get the headline of the text 
    
    Args:
        history: chat history 
  """
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
  
# Function for handling inputs into the generative AI model
def transform(input: str, history: list[mel.ChatMessage]):
    state = me.state(State)
    chat_history = ""
    if state.file and state.uploaded:
       chat_history += f"\nuser: {pdf_to_text(state.file)}"
    # Creating the chat_history
    chat_history += "\n".join(f"{message.role}: {message.content}" for message in history)
    # implementation with chromaDB goes here
    # results = collection.query(query_texts=[state.article_title],
    #                                  n_results=3,
    #                                  where=
    #                                  {
    #                                     "label": "true"
    #                                  })
    # chromadb_info = "\n".join(results['documents'][0])


    # full_input = f"{chat_history}\nChromaDB Info: Based on the headline, these are the most similar true statements: {chromadb_info}\nuser: {input}"
    # time.sleep(4)

    # full_input = f"{chat_history}\nChromaDB Info: Based on the headline, these are the most similar true statements: {chromadb_info}\nuser: {input}"
    # Combining input and chat_history
    full_input = f"{chat_history}\nuser: {input}"
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
    # normal prompt get score
    overall_sens_normal_prompt_match = re.search(r'normal\s*prompting\s*overall\s*sensationalism\s*:\s*(\d+(\.\d+)?)', text_chunk, re.IGNORECASE)
    overall_stance_normal_prompt_match = re.search(r'normal\s*prompting\s*overall\s*stance\s*:\s*(\d+(\.\d+)?)', text_chunk, re.IGNORECASE)
    if overall_sens_normal_prompt_match:
        state.overall_sens_normal_score = float(overall_sens_normal_prompt_match.group(1))
    if overall_stance_normal_prompt_match:
        state.overall_stance_normal_score = float(overall_stance_normal_prompt_match.group(1))
    # fcot prompt get score
    overall_sens_fcot_prompt_match = re.search(r'fcot\s*prompting\s*overall\s*sensationalism\s*:\s*(\d+(\.\d+)?)', text_chunk, re.IGNORECASE)
    overall_stance_fcot_prompt_match = re.search(r'fcot\s*prompting\s*overall\s*stance\s*:\s*(\d+(\.\d+)?)', text_chunk, re.IGNORECASE)
    if overall_sens_fcot_prompt_match:
        state.overall_sens_fcot_score = float(overall_sens_fcot_prompt_match.group(1))
    if overall_stance_fcot_prompt_match:
        state.overall_stance_fcot_score = float(overall_stance_fcot_prompt_match.group(1))
    print('checking for bug')
    print(state.overall_sens_fcot_score, state.overall_stance_fcot_score)
    state.normal_prompt_vs_fcot_prompt_log['normal_prompt'] = ["sensationalism: " + str(round(float(state.overall_sens_normal_score),2)),
                                                         "political_stance: " + str(round(float(state.overall_stance_normal_score),2))]
    state.normal_prompt_vs_fcot_prompt_log['fcot_prompt'] = ["sensationalism: " + str(round(float(state.overall_sens_fcot_score),2)),
                                                             "political_stance: " + str(round(float(state.overall_stance_fcot_score), 2))]
    print("####### prompt log ########")
    print(state.normal_prompt_vs_fcot_prompt_log)
    # FIX bug where if model is asked questions. veracity will automatically populate
    state.veracity = round(np.mean([state.overall_naive_realism_score, 10 - state.overall_sens_score, state.overall_stance_score, state.overall_social_credibility]), 2)

def transform_test(input: str, history: list[mel.ChatMessage]):
  state = me.state(State)
  chat_history = ""
  if state.file and state.uploaded:
    chat_history += f"\nuser: {pdf_to_text(state.file)}"
  # Creating the chat_history
  chat_history += "\n".join(f"{message.role}: {message.content}" for message in history)
  full_input = f"{chat_history}\nuser: {input}"
  time.sleep(4)
  response = model.generate_content(full_input, stream=True)
  for chunk in response:
    yield chunk.text

def transform_fc(input: str, history: list[mel.ChatMessage]):
  state = me.state(State)
  chat_history = ""
  if state.file and state.uploaded:
      chat_history += f"\nuser: {pdf_to_text(state.file)}"
  chat_history += "\n".join(f"{message.role}: {message.content}" for message in history)
