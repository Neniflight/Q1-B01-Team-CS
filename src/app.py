
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
import matplotlib.pyplot as plt 
from questions import predefined_questions
from normal_prompting import normal_prompting_question
from fcot_prompting import fcot_prompting_question
from naive_realism import naive_realism_normal, naive_realism_cot, naive_realism_fcot

from serpapi import GoogleSearch
from dotenv import load_dotenv
import os
from newspaper import Article
from newspaper import ArticleException
from datetime import datetime, timedelta
import pdfkit

# from the env file, get your API_KEY
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_collection("Misinformation")

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
  normal_prompting_question: dict[str, str] = field(default_factory=lambda: normal_prompting_question)
  cot_prompting_question: dict[str, str] = field(default_factory=lambda: normal_prompting_question)
  fcot_prompting_question: dict[str, str] = field(default_factory=lambda: fcot_prompting_question)
  chat_history: list[mel.ChatMessage]
  overall_naive_realism_score: float = 0.0
  veracity: float = 0.0
  veracity_label: str = ""
  # overall_sens_normal_score: float = 0.0
  # overall_stance_normal_score: float = 0.0
  # overall_sens_cot_score: float = 0.0
  # overall_stance_cot_score: float = 0.0
  # overall_sens_fcot_score: float = 0.0
  # overall_stance_fcot_score: float = 0.0
  normal_prompt_vs_fcot_prompt_log: dict[str, str] = field(default_factory=dict)
  normal_response_dict: dict[str, str] = field(default_factory=dict)
  cot_response_dict: dict[str, str] = field(default_factory=dict)
  fcot_response_dict: dict[str, str] = field(default_factory=dict)
  link: str = ""
  finish_analysis: bool = False
  vdb_response: str = ""
  serp_response: str = ""
  test_response: str = ""
  selected_values_1: list[str] = field(default_factory=lambda: [])
  radio_value: str = ""
  toggle_values: list[str] = field(default_factory=lambda: [])
  response: str = ""
  article_author: str = ""
  article_date: str = ""
  article_source: str = ""
  article_topic: str = ""
  article_summary: str = ""


  # citation for using dict: https://github.com/google/mesop/issues/814

# Function used to handle when a user uploads a pdf file
def handle_upload(event: me.UploadEvent):
  state = me.state(State)
  state.file = event.file
  state.uploaded = True
  print('before get_metadata')
  get_metadata(state.chat_history)
  print('after get_metadata')

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

def convert_url_to_pdf(url, pdf_path):
  path_wkhtmltopdf = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe" # install using https://wkhtmltopdf.org/ and input proper path
  config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
  try:
    pdfkit.from_url(url, pdf_path, configuration=config)
    print(f"PDF generated and saved at {pdf_path}")
  except Exception as e:
    print(f"PDF generation failed: {e}")
  # convert_url_to_pdf("https://www.cnn.com/2025/02/25/politics/trump-court-losses-immigration-spending-freeze/index.html", r"C:\temp\example.pdf")

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
  # checking user selection to determeine if we are adding new info to our prompt 
  user_article_title = state.article_title
  input = ""
  if "Vector_Database" in state.toggle_values:
    # Connecting to the vector database to allow for more informed prompts
    vdb_results = collection.query(query_texts=[user_article_title], n_results=3)
    vdb_results = str(vdb_results['metadatas'])
    vdb_str = f"ChromaDB Info: Based on the headline, these are the most similar statements: {vdb_results}"
    state.vdb_response = vdb_str
    input = input + vdb_str + "\n"
    print("added vector database info to normal question")
  if "SERP_API" in state.toggle_values:
    articles_from_serp_api = str(serp_api(user_article_title))
    print(user_article_title)
    serp_str = f"SERP API: These are similar articles found online via an API. Please consider these articles' information in the score: {articles_from_serp_api}"
    state.serp_response = serp_str
    input = input + serp_str + "\n"
    print("added serp_api info to normal question")
  selected_normal_keys = state.selected_values_1
  state.response = ''
  for i in state.normal_prompting_question.keys():
    # print(f"Question:{question}")
    if i in selected_normal_keys:
      print('start asking normal prompting question')
      print(i)
      response_generator = transform(state.normal_prompting_question[i] + input, state.chat_history)  
      state.response = ''.join(response_generator)
      print(f"Response:{state.response}")
      time.sleep(5)
      state.normal_response_dict[i] = state.response
      overall_sens_match = re.search(r'overall\s*sensationalism\s*:\s*(\d+(\.\d+)?)', state.response, re.IGNORECASE)
      overall_stance_match = re.search(r'overall\s*stance\s*:\s*(\d+(\.\d+)?)', state.response, re.IGNORECASE)
      if overall_sens_match:
        print('found_sens')
        state.overall_sens_score = float(overall_sens_match.group(1))
      if overall_stance_match:
        print('found_stance')
        state.overall_stance_score = float(overall_stance_match.group(1))
  print(state.normal_response_dict)

def ask_cot_prompting_questions(event: me.ClickEvent):
  """loop through our cot prompted questions to ask gemini to give us a score of 1 to 10 
    for the sensationalism and political stance
    
    Args:
        event: this question is activated when the button associated with this function is clicked 
  """
  state = me.state(State)
  # checking user selection to determeine if we are adding new info to our prompt 
  user_article_title = state.article_title
  input = ""
  if "Vector_Database" in state.toggle_values:
    # Connecting to the vector database to allow for more informed prompts
    vdb_results = collection.query(query_texts=[user_article_title], n_results=3)
    vdb_results = str(vdb_results['metadatas'])
    vdb_str = f"ChromaDB Info: Based on the headline, these are the most similar statements: {vdb_results}"
    state.vdb_response = vdb_str
    input = input + vdb_str + "\n"
    print("added vector database info to cot question")
  if "SERP_API" in state.toggle_values:
    articles_from_serp_api = str(serp_api(user_article_title))
    print(user_article_title)
    serp_str = f"SERP API: These are similar articles found online via an API. Please consider these articles' information in the score: {articles_from_serp_api}"
    state.serp_response = serp_str
    input = input + serp_str + "\n"
    print("added serp_api info to cot question")
  cot_keys = state.selected_values_1
  state.response = ''
  for i in state.cot_prompting_question.keys():
    # print(f"Question:{question}")
    if i in cot_keys:
      print('start asking cot prompting question')
      print(i)
      response_generator = transform(state.cot_prompting_question[i] + input, state.chat_history)  
      state.response = ''.join(response_generator)
      print(f"Response:{state.response}")
      time.sleep(5)
      state.cot_response_dict[i] = state.response
      overall_sens_match = re.search(r'overall\s*sensationalism\s*:\s*(\d+(\.\d+)?)', state.response, re.IGNORECASE)
      overall_stance_match = re.search(r'overall\s*stance\s*:\s*(\d+(\.\d+)?)', state.response, re.IGNORECASE)
      if overall_sens_match:
        print('found_sens')
        state.overall_sens_score = float(overall_sens_match.group(1))
      if overall_stance_match:
        print('found_stance')
        state.overall_stance_score = float(overall_stance_match.group(1))
  print(state.cot_response_dict)

def ask_fcot_prompting_questions(event: me.ClickEvent):
  """loop through our fractal chain of thought prompted questions (3 iterations) to ask gemini to give us a score of 1 to 10 
    for the sensationalism and political stance
    
    Args:
        event: this question is activated when the button associated with this function is clicked 
  """
  state = me.state(State)
  # checking user selection to determeine if we are adding new info to our prompt 
  user_article_title = state.article_title
  input = ""
  if "Vector_Database" in state.toggle_values:
    # Connecting to the vector database to allow for more informed prompts
    vdb_results = collection.query(query_texts=[user_article_title], n_results=3)
    vdb_results = str(vdb_results['metadatas'])
    vdb_str = f"ChromaDB Info: Based on the headline, these are the most similar statements: {vdb_results}"
    state.vdb_response = vdb_str
    input = input + vdb_str + "\n"
    print("added vector database info to fcot question")
  if "SERP_API" in state.toggle_values:
    articles_from_serp_api = str(serp_api(user_article_title))
    print(user_article_title)
    serp_str = f"SERP API: These are similar articles found online via an API. Please consider these articles' information in the score: {articles_from_serp_api}"
    state.serp_response = serp_str
    input = input + serp_str + "\n"
    print("added serp_api info to fcot question")
  # looping through fcot selection to ask user selected questions in our fcot_prompts
  fcot_keys = state.selected_values_1
  for i in state.fcot_prompting_question.keys():
    if i in fcot_keys:
      print("start asking fcot prompting questions")
      response_generator = transform(state.fcot_prompting_question[i] + input, state.chat_history)  
      state.response = ''.join(response_generator)
      print(f"Response:{state.response}")
      time.sleep(5)
      state.fcot_response_dict[i] = state.response 
      overall_sens_match = re.search(r'overall\s*sensationalism\s*:\s*(\d+(\.\d+)?)', state.response, re.IGNORECASE)
      overall_stance_match = re.search(r'overall\s*stance\s*:\s*(\d+(\.\d+)?)', state.response, re.IGNORECASE)
      if overall_sens_match:
        print('found_sens')
        state.overall_sens_score = float(overall_sens_match.group(1))
      if overall_stance_match:
        print('found_stance')
        state.overall_stance_score = float(overall_stance_match.group(1))
  print(state.fcot_response_dict)

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
    # Connecting to the vector database to allow for more informed prompts
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

    
def ask_pred_ai():
  """Runs two Predictive AI models (sentiment_analyzer & social_credit_model) and returns a score for each factuality factors
    
    Args:
        event: this question is activated when the button associated with this function is clicked 
  """
  state = me.state(State)
  if "Naive_realism" in state.selected_values_1:
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
      'barely_true_ratio': 0 if speaker_reput is None else speaker_reput[0],
      'false_ratio': 0 if speaker_reput is None else speaker_reput[1],
      'half_true_ratio': 0 if speaker_reput is None else speaker_reput[2],
      'mostly_true_ratio': 0 if speaker_reput is None else speaker_reput[3],
      'pants_on_fire_ratio': 0 if speaker_reput is None else speaker_reput[4],
      'confidence': confidence,
      'subjectivity': subject_score,
      }, index=[0])
    # {'barely-true': 0, 'false': 1, 'half-true': 2, 'mostly-true': 3, 'pants-fire': 4, 'true': 5}
    prediction = loaded_model.predict(df.loc[0:0])[0]
    ### LINE ABOVE IS GIVING ERROR COMMENTING OUT TO WORK ON SCORE TABLE ###
    # prediction = 5
    prediction_to_score = {5: 6, 4: 1, 0: 3, 1: 2, 2: 4, 3: 5}
    state.overall_naive_realism_score = prediction_to_score[prediction]
  
  if "Social_credibility" in state.selected_values_1:
    # load model
    social_credit_model = speaker_context_party_nn()
    state_dict = torch.load("../model/speaker_context_party_model_state.pth")
    social_credit_model.load_state_dict(state_dict)
    print("hello loaded model!")
    # citation: https://discuss.pytorch.org/t/error-loading-saved-model/8371/6

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
    speaker_generator = transform("Who is the speaker in this article? Only give the speaker name", state.chat_history)
    speaker = ''.join(speaker_generator)
    speaker = ('-'.join(speaker.split())).lower()
    speaker = speaker.replace("\n", "").replace(" ", "")
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
      speaker_score = modified_label['mod_label'].mean()
      context_score = modified_label['mod_label'].mean()
      party_affli_score = modified_label['mod_label'].mean()
      state.overall_social_credibility = (speaker_score + context_score + party_affli_score) / 3
    state.overall_social_credibility = state.overall_social_credibility * 6 / 10
    print(f"This is overall_social_credibility: {state.overall_social_credibility}")
    state.overall_social_credibility = round(state.overall_social_credibility, 2)
  
  print(state.overall_social_credibility)

  all_scores = [state.overall_naive_realism_score, 6 - state.overall_sens_score, 6 - state.overall_stance_score, state.overall_social_credibility]
  exist_score_sum = []
  for scores in all_scores:
    if scores:
      exist_score_sum.append(scores)
  state.veracity = np.round(np.mean(exist_score_sum), 2)
  

  # state.veracity = round(np.mean([state.overall_naive_realism_score, 10 - state.overall_sens_score, state.overall_stance_score, state.overall_social_credibility]), 2)
  # the label is assigned here but i have issues 
  if state.veracity >= 0 and state.veracity < 1:
    label = "Pants on Fire"
  elif state.veracity >= 1 and state.veracity < 2:
    label = "False"
  elif state.veracity >= 2 and state.veracity < 3:
    label = "Barely True"
  elif state.veracity >= 3 and state.veracity < 4:
    label = "Half True"
  elif state.veracity >= 4 and state.veracity < 5:
    label = "Mostly True"
  else: 
    label = "True"
  #pants on fire, barely true, half true, mostly true, true, false
  state.veracity_label = label

def navigate_to(event: me.ClickEvent, path: str):
  me.navigate(path)

navigate_to_ve = partial(navigate_to, path="/Gemini_Misinformation_ChatBot")
navigate_to_normal = partial(navigate_to, path="/normal_adjustments")
navigate_to_cot = partial(navigate_to, path="/cot_adjustments")
navigate_to_fcot = partial(navigate_to, path="/fcot_adjustments")
navigate_to_about = partial(navigate_to, path="/about_us")

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
  with me.box(style=me.Style(width="100%", height="100vh", background="white", flex_direction="column", justify_content="flex-start", display="flex", margin=me.Margin.all(0), overflow="auto")):
    # nav bar
    with me.box(style=me.Style(position='fixed', width="100%", display='flex', top=0, overflow='hidden', justify_content="space-between", align_content="center", background='white', border=me.Border(bottom=me.BorderSide(width="0.5px", color='#010021', style='solid')), padding=me.Padding.symmetric(vertical=15, horizontal=50), z_index=10)):
      me.html(
        """
        <a href="/">
          <img src="https://res.cloudinary.com/dd7kwlela/image/upload/v1738889378/capstone-dsc180b/jiz38dkxevducq0rpeye.png" alt="Home" height=48>
        </a>
        """
      )
      with me.box(style=me.Style(justify_content="flex-start", align_items="center", gap=40, display="flex")):
        me.link(text="Try Chenly Insights", url="/insights", style=me.Style(text_decoration='none', font_family='Inter', color="white", font_size=16, font_weight='bold', background="#010021", padding=me.Padding.symmetric(vertical=8, horizontal=10), border_radius=5))
        me.link(text="Prompt Testing", url="/prompt_testing", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
        me.link(text="Pipeline Explanation", url="/pipeline_explanation", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
        me.link(text="About Us", url="/about_us", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
    # title card
    with me.box(style=me.Style(align_self="stretch", justify_content="center", display='flex')):
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
    # examples section
    with me.box(style=me.Style(align_self="stretch", background="linear-gradient(90deg, #5271FF 0%, #22BB7C 100%)", justify_content="center", display='flex')):
      with me.box(style=me.Style(width="100%", padding=me.Padding.all(100), max_width=1440, flex_direction="column", justify_content="flex-start", align_items= "flex-start", gap=20, display="inline-flex")):
        me.text("Samples", type="headline-3", style=me.Style(align_self="stretch", color="white", font_family="Inter", font_weight=700, word_wrap="break-word", margin=me.Margin.all(0)))
        with me.box(style=me.Style(justify_content="center", align_items="flex-start", gap=30, display="inline-flex")):
          me.button("Trusted News Articles", type="stroked", style=me.Style(color="white", font_family="Inter", font_weight="bold", font_size="20px", border=me.Border.all(me.BorderSide(width="1.5px", color="white", style="solid"))))
          me.button("Satirical Articles", type="stroked", style=me.Style(color="white", font_family="Inter", font_weight="bold", font_size="20px", border=me.Border.all(me.BorderSide(width="1.5px", color="white", style="solid"))))
          me.button("Sketchy Sources", type="stroked", style=me.Style(color="white", font_family="Inter", font_weight="bold", font_size="20px", border=me.Border.all(me.BorderSide(width="1.5px", color="white", style="solid"))))
        with me.box(style=me.Style(display="flex", align_self="stretch", justify_content="center")):
          me.image(src="https://archive.org/download/placeholder-image/placeholder-image.jpg", style=me.Style(align_self="stretch", border_radius="10px", height="572px"))
    with me.box(style=me.Style(align_self="stretch", justify_content="center", display='flex')):
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
            me.text("This project was done for DSC 180 Capstone for the Winter 2025 Showcase. We collaborated with Dr. Ali Arsanjani, a Director at Google AI, on the subject “GenAI for Good”, where we fight against misinformation present in our society with AI. As a group of two, we present the Chenly Insights. A cutting edge technology that uses the latest generative and predictive AI to fight against misinformation in news article. You can upload or link different news articles and received detail breakdowns of its truthfulness depending on different factuality factors. You can view more information about our tool and its paper through the “Learn More” button. ", type="body-1", style=me.Style(font_family="Inter", color="#010021", word_wrap="break-word", width=650))
            me.image(src="https://res.cloudinary.com/dd7kwlela/image/upload/v1738889378/capstone-dsc180b/jiz38dkxevducq0rpeye.png", style=me.Style(margin=me.Margin(left=30), width=430, height='auto'))
          me.button("Learn More", type='flat', style=me.Style(font_family="Inter", font_size="20px", font_weight="bold"), on_click=navigate_to_about)
        with me.box(style=me.Style(align_self="stretch", justify_content="space-between", align_items="flex-start", gap=150, display="inline-flex")):
          with me.card(appearance="outlined", style=me.Style(font_family="Inter", border=me.Border.all(me.BorderSide(width="2px", color="#5271FF", style="solid")))):
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
          
          with me.card(appearance="outlined", style=me.Style(font_family="Inter",border=me.Border.all(me.BorderSide(width="2px", color="#5271FF", style="solid")))):
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

# About us page 
@me.page(path="/pipeline_explanation", stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap"
    ])
def pipeline_explanation():
  """
  contains an explanation of how the pipeline works
  """
  with me.box(style=me.Style(background="white", width="100%", display="flex", flex_direction="column", justify_content="flex-start", margin=me.Margin.all(0), overflow="auto")):
    # nav bar
    with me.box(style=me.Style(position='fixed', width="100%", display='flex', top=0, overflow='hidden', justify_content="space-between", align_content="center", background='white', border=me.Border(bottom=me.BorderSide(width="0.5px", color='#010021', style='solid')), padding=me.Padding.symmetric(vertical=15, horizontal=50), z_index=10)):
      me.html(
        """
        <a href="/">
          <img src="https://res.cloudinary.com/dd7kwlela/image/upload/v1738889378/capstone-dsc180b/jiz38dkxevducq0rpeye.png" alt="Home" height=48>
        </a>
        """
      )
      with me.box(style=me.Style(justify_content="flex-start", align_items="center", gap=40, display="flex")):
        me.link(text="Try Chenly Insights", url="/insights", style=me.Style(text_decoration='none', font_family='Inter', color="white", font_size=16, font_weight='bold', background="#010021", padding=me.Padding.symmetric(vertical=8, horizontal=10), border_radius=5))
        me.link(text="Prompt Testing", url="/prompt_testing", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
        me.link(text="Pipeline Explanation", url="/pipeline_explanation", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
        me.link(text="About Us", url="/about_us", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
  # header
    with me.box(style=me.Style(align_self="stretch", justify_content="center", display='flex', background= "linear-gradient(to right, #5271FF , #22BB7C)")):
      with me.box(style=me.Style(width="100%", max_width=1440, height="auto", padding = me.Padding.symmetric(vertical=100, horizontal=100),margin = me.Margin(top=80, bottom=10))):
          me.text(text="Pipeline Explanation", type="headline-2", 
                style = me.Style(font_weight = "bold", color ="white", font_family = "Inter", margin=me.Margin.all(0)))
    with me.box(style=me.Style(align_self="stretch", background="white", justify_content="center", display="flex")):
      with me.box(style=me.Style(width="100%", padding=me.Padding(top=100, right=100, left=100, bottom=20), max_width=1440, flex_direction="column", justify_content="flex-start", align_items= "flex-start", gap=20, display="inline-flex")):
        me.text(text = "Pipeline", type="headline-3", style = me.Style(font_weight = "bold", color ="010021", font_family = "Inter", margin=me.Margin.all(0)))
        me.text(text = "Our pipeline consists the Liar Plus Dataset, ChromaDB, Articles online, Predictive AI, and Generative AI that work together to come up with a veracity score for an user inputted article. A user will upload an article that they are interested to see whether there is misinformation. It is fed into both predictive AI and generative AI. Each type of AI has their own models and factuality factors to judge the article off of. The generative AI also is connected to a Docker server that contains a ChromaDB image of similar statements to the headline that are true. All of these factors work together to come up with a veracity score.", type="body-1", style=me.Style(font_family="Inter"))
    with me.box(style=me.Style(align_self="stretch", background="white", justify_content="center", display="flex")):
      with me.box(style=me.Style(width="100%", padding=me.Padding(top=100, right=100, left=100, bottom=20), max_width=1440, flex_direction="column", justify_content="flex-start", align_items= "flex-start", gap=20, display="inline-flex")):
        me.text(text = "LucidChart", type="headline-3", style = me.Style(font_weight = "bold", color ="010021", font_family = "Inter", margin=me.Margin.all(0)))
        me.text(text = "If you would like to view the entirety of the pipeline, we have visualized it on LucidChart embed below! ", type="body-1", style=me.Style( font_family="Inter"))
        me.embed(src="https://lucid.app/documents/embedded/7babda6c-da85-49d9-bb0c-83fd85deffdf", style=me.Style(width="960px", height="720px"),)


# About us page
@me.page(path = "/about_us",stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap"
    ])
def about_us():
  """
  contains general description, techonologies, about the members, and research paper.
  """
  with me.box(style=me.Style(background="white", width="100%", display="flex", flex_direction="column", justify_content="flex-start", margin=me.Margin.all(0), overflow="auto")):
    # nav bar
    with me.box(style=me.Style(position='fixed', width="100%", display='flex', top=0, overflow='hidden', justify_content="space-between", align_content="center", background='white', border=me.Border(bottom=me.BorderSide(width="0.5px", color='#010021', style='solid')), padding=me.Padding.symmetric(vertical=15, horizontal=50), z_index=10)):
      me.html(
        """
        <a href="/">
          <img src="https://res.cloudinary.com/dd7kwlela/image/upload/v1738889378/capstone-dsc180b/jiz38dkxevducq0rpeye.png" alt="Home" height=48>
        </a>
        """
      )
      with me.box(style=me.Style(justify_content="flex-start", align_items="center", gap=40, display="flex")):
        me.link(text="Try Chenly Insights", url="/insights", style=me.Style(text_decoration='none', font_family='Inter', color="white", font_size=16, font_weight='bold', background="#010021", padding=me.Padding.symmetric(vertical=8, horizontal=10), border_radius=5))
        me.link(text="Prompt Testing", url="/prompt_testing", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
        me.link(text="Pipeline Explanation", url="/pipeline_explanation", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
        me.link(text="About Us", url="/about_us", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
  # header
    with me.box(style=me.Style(align_self="stretch", justify_content="center", display='flex', background= "linear-gradient(to right, #5271FF , #22BB7C)")):
      with me.box(style=me.Style(width="100%", max_width=1440, height="auto", padding = me.Padding.symmetric(vertical=100, horizontal=100),margin = me.Margin(top=80, bottom=10))):
          me.text(text="About Us", type="headline-2", 
                style = me.Style(font_weight = "bold", color ="white", font_family = "Inter", margin=me.Margin.all(0)))
  # general description section
    with me.box(style=me.Style(align_self="stretch", background="white", justify_content="center", display="flex")):
      with me.box(style=me.Style(width="100%", padding=me.Padding(top=100, right=100, left=100, bottom=20), max_width=1440, flex_direction="column", justify_content="flex-start", align_items= "flex-start", gap=20, display="inline-flex")):
        me.text(text = "General Description", type="headline-3", style = me.Style(font_weight = "bold", color ="010021", font_family = "Inter", margin=me.Margin.all(0)))
        with me.box(style=me.Style(align_self="stretch", justify_content='space-between', align_items='flex-start', display='inline-flex')):
          me.text(text = "This project was done for DSC 180 Capstone for the Winter 2025 Showcase. We collaborated with Dr. Ali Arsanjani, a Director at Google AI, on the subject “GenAI for Good”, where we fight against misinformation present in our society with AI. As a group of two, we present the Chenly Insights. A cutting edge technology that uses the latest generative and predictive AI to fight against misinformation in news article. You can upload or link different news articles and received detail breakdowns of its truthfulness depending on different factuality factors.", type="body-1", style=me.Style(width=650, font_family="Inter"))
          me.image(style=me.Style(width="430px"),
            src="data:image/png;base64,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"
          )
  # techonology section
    with me.box(style=me.Style(align_self="stretch", background="white", justify_content="center", display="flex")):
      with me.box(style=me.Style(width="100%", padding=me.Padding.symmetric(horizontal=100, vertical=20), max_width=1440, flex_direction="column", justify_content="flex-start", align_items= "flex-start", gap=20, display="inline-flex")):
        me.text(text="Technologies", type="headline-3", style = me.Style(font_weight = "bold", color ="010021", font_family = "Inter", margin=me.Margin.all(0)))
        with me.box(style=me.Style(align_self='stretch', display='flex', justify_content="space-between", align_items="flex-start")):
          with me.box(style=me.Style(width=650, display="flex", flex_direction="column", align_content="flex-start", gap=0)):
            me.text("The following technologies were used for our project: ", type="body-1", style=me.Style(font_family="Inter", margin=me.Margin.all(0)))
            me.markdown("""
*   Google Gemini
*   Interface: Mesop
*   Languages: Python
*   Packages: Pytorch, Textblob, Transformers, Newspaper3k, Numpy, XGBoost
*   Vector Database: ChromaDB
*   Search Engine: SERP API
*   Host: Docker
              """, style=me.Style(font_family="Inter", font_size="16px", color="#010021"))
          with me.box(style=me.Style(justify_content="flex-start", align_content="center", flex_wrap="wrap", display='flex', width="40%", gap=66)):
            me.image(style=me.Style(height=75),
              src="data:image/png;base64,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"
            )
            me.image(style=me.Style(height=75),
              src="data:image/png;base64,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"
            )
            me.image(style=me.Style(height=75),
              src="data:image/png;base64,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"
            )
            me.image(style=me.Style(height=75),
              src="data:image/png;base64,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"
            )
            me.image(style=me.Style(width=75),
              src="data:image/png;base64,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"
            )
            me.image(style=me.Style(width=75),
              src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJgAAACWCAYAAAAiyEFRAAAACXBIWXMAABYlAAAWJQFJUiTwAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAE6MSURBVHgB7b0JmB3FeS78VXefc2bO7KtmNNrQBggEYrXB7MZg8A7GS2y8x7HvjW/sxP6Te//fv//Eefw/uU5uEju24zje8UoAGzDYGMwOBgyITSAhtEsjzabZl3NOd92q6qrqr6qrz5wR2p7npvSMTndXV9VXVe+31FdLEzjygcg/Kv+MsH79X7UtKW44salpw7rF7Seu9YtjKyvlaBmE0EsItLBX6iolyA/1V8jZFxTgA59uhK5eL7Owvdsr8KN/nIIXHi/DosU+eF5cOpGlEyJfpDFRgkBKYsqIfGbEoXQ0qRDOw77X13aeNHmZRK44YjSaygdwuviHpaYlSsksux+lhPbTILezc7Z+2/Do5s2vzP1h0+bJp7bet+OfRiEdZIvwPNL9cbhDAEcu8ErwykTyT4QvfOEL3l2/GO9tbjplfUPT4lNagsVr6uoXrSBevjck421RKWoESguEkByrvc/7gHi6HxZMAZXgorIzKQKZ6DcRR1kcMTpSXep0kIBMv6buEZCMOJyRmb2ZL8rLLh9wXslz1q6kwJgnF0XQwMjvhnLlxDFvajJo6x05Ibqyv7359O3ret+wZby8b9OeoS3P3bfjy/tlFhHK1YcjDLQjATBFeCj/RFi//uttyxadcOnw1g2vv/TczguKDcFaFlk3OwtQLlcgrMyw3xmgUcCkDkk6X3YoNURF9aCYXnei+KXsl2SCBQNQSTwcp7vAFYfzqJYnB1kky0YyXechKxunIwadilgqkUzi130Wz9qa5AiNiiVSaQ+C+mVNftOGxU3dkC+cxpE4Md5Ht7x+1V88ND70zN2DY5se+t6Oz4zKXENJSSCvDzvQDifAMLAq6uHll39z2bKuq65rb1369oYCnMOkUWFqisL+4XHW2CXRgiRGD+XtxOJTHcQDVSUsIGiQCaCSuINJAlxDkgkAxh0LYL6nwKGa3waSkE40G2QASD0SR/1owkAEvasxjbhC5Q1gNQdrOI96QMMyzLC/aU7QNECO5Jua6opnrepYdFZz+xUfnZm54vHVS99xy1NDv7np5hf/pF/mzPvLg0TjHDagHS6AcWDx+mpgXXbZN/qW97zlw4s6+q5vLMLa6akIRg5OMkle5m1BiehwYQ4QoaKolDMksXuUilOB1lhtSkypY6rHDEkmAafLBxPgrjJqBRlWe1AFgAYYUyAjyE5ULGHSR0jSRkS+WKElGJmdI2SGQH9QaGyuq7/spL7ll61e9PGPr++98gfbhu78wQ+f/eQAJKozAMuseTXh1QJMiVcOLFG1q1bfUWjfcNZHV6/q/jMGn7VjoxHsHh0BTi+T5hEhUkGoDpG9gjvWZa8cEmXEuocqag8cINMvmxKU2JKMWmoPLFBDun7gAKCiT+m/GHQkLclQgkyQKU0qmCkGW1iZg5HJaTI+FUB7oWX9hs7lXz6x4xPXn9v89n96YOTBH9246V1MpWhpdljU5qsBGCeCS66yevCet91//pJlF32xvgCXTU5GMDF5kKnBiHrEZ9zAx3PUo0AN+8IAEiScTDQAZZwNmKyggCTzwXnGHUecgEgASOXjKvYaQMoAt4FrkIqkMaFVJBlNSzVMoi6PWmVCGmSa3iSSeAKsATMDaTQ0O+yPzOWgpa75tBMX9Xynt/O6a1a03vv5Lz9y6UZIJFge4v49ZJB5cGjBl2kFuNat+3n+zz459f+cdMpFv80xcA2MDMHo+DDXRRXP93lf+XFdJXIADKmgVR+huqF1nGo5Unst7Q6htiQj6XexYKDUTGfE4Txp8q5RrqMOupwMqYrvDbq5JJNvUKuS2B5LtY0sk1qRUrD5xAu4q6NycHoINk6MwUwO3nze8kvu+crbxz79BfiCwgWXaFwIHSpOFpxQqURlGMKbL//tsrdddc2t7R3FLx4YKhX3H+iHKAQFLM4unk4pA5ZGVFKRNDJNd6a+MS6cwehIS+Uana5eJg6QITVjp0u1BjgACG5wGjTKPCkGtSMdtSQWplGiJRNkSt3bJobELdMmEPheLuRDg/6hQRg8WGlfWd/8jyuv/f9+9tnTftMtXy/LJD4cQlgIwBS4uF4WBuB73rPvwnVnXv5wifpX7tp7EObmRmng5SuMdiHhKOosuxFcdlXSyDRVspPLMwIuw0jnko4aSEhKiHdld2JphWlBatcGYOo9fG/Rgu9TbQQIIJQapgRuByzJUkFJMgq2JBOBdaTP3IyeF+Qq4+UxumVwkjnX4J1Xr7riwX84d8/p8jVliy3YpKoVYApc2pj/4Ad3vffkk3t/HYZ0yTBTiVFUrnCxG3kk4OaLwZEEXUM2t2J1qUCG/WA1G/tIrYKlnpwSAqdJAYIaatVORx0AxHQsGJyW5DTLSJiAWvVUIEtJMUQHzhOS6oJwzRESMJSx7pwL+ydHYY8Ha9eu7rvvCxdtvlq+quyyBYGsVoBxiZSA68N7Prlk2dIbBodpcWhoP3eMlj0/UEY/0BRYUG2qAMkZ5wBALcEAEpjSwykpuKXjsgGN/JIOtiWSbVu5pBUGpymdTCYz8nSAjGiazfZJnpuBoLxdzYcAG7B+JBDR8r7RARiagtazF6+95QuXbvkj+eqCQVYLwIzphPd9aPen+pb2fX10jHrj4xxcuTJzYQVUqURFNG4sgPmlFbHirHSa/WqQYhT9aWkBVqfjOIIiBC3UeJ+q7kRzhS7g4rJTdXBJK/VehpRzARcQyECZ/wj81SSZaySOH8k0zGPrBczUKQ1MDsLQDOTP7l7zg7+55OX3y9cUyGqyybwa44XNdf1Hdn902fIl/8zABeNjDFx+rswRllKJkFzjilIvqUjS6PEFBoEJCBpLFoilTM2BgCk9bImG4jR9qkTle5KdrOkCE/DOPMF6ZpcH6c6vpoJt1U0RyDBdxkQ5dZejmpu6Io28hPcs73uBApl/Rvfq737xoi3XyFflVP38AqraCzwDNfUD7/3QnrcsWb7kX8fY3L2QXAEDF3i5lGgHJNqRPeHqBMM2AKviFlspcLoaJisYHQLZIKMaLCQBMZKkCc1KktHqeaL6ZcVVTaeeUZPOhALVVpYLw2YqqAIyR3lGkwuV6+V94pcHpgZheBaCM5es+cHnL952oXyNm0w+zKNTsgDGE+VA+rne+Na7T1uxsu/74+MQjDFwsaEtn+/JgUUQBkjVYXpWHFIX+L5aJ2RR7wQruq42CFHUOQFBHe9mgIW+2nReFXBmtQWSfPOpyyxJph/zPIif80lQOsBAdnAGGs7vOeGnnzz79pXyNY6PqvZYFsA4MrmTDc4779/bV6+77EcTE9A2NjoIPlOLVIIrJX0s0Z5lMOv3wRGHpYq6N/m2pqC4lOJGpianZ0lOriqpQ82k7EqUZ+agAiBz0IMZDMel1KuWnJCqHyA6kzZM7qqqSzDzsEMCMk+AbP/EKIyUYfE5i6/+/sd7v1mUrylJ5gxexjM90XnKOR/6+/oiOXVsdIgXySVXgBvLthFssKQkEljpIJ0uPbqSqotaCTOCDVatqi3A47INFUREq6ZpNu4p6JkHXB6AMSIGixYXyKgFeLsttDEPJgBxnTRgIF0muOIA9FRdFsh0GrEugfugKpX9kxPQXiQXXL7+w19Er2XKVBfA1KgRrr5uz/Ut7f6HhwemIaSVkPiBjwYwmTaC+KVmJXEFzcZDz1KNR804glPXGDDIM0CmaElLsvQ1ppNmxOF4F5Bwmfa10W4GE5MUU+v5VqkLE5AhBpGFZ0oyXJfqTesFfo5UKrN093gFmttyn/qnc7e8WcZlSjHPcS/srosu+snS1Sf1fWl6JoKZmTFucrH6UE+rHYvrXMNpJ5BsIGaAzAQATQNwnkDBAkEtqptASn1VA5m+Q72Xko4A2VIcMtpNZedZ5SEGM8CqQebw9tughgyQkTTIjLziMv0cm6mZnhuBvWXIrVy65u8+cNL3O2Q0HwymBJaXrl4cVp1+xf9NPFgyevAgmxrNVYgy5rBayACL0ciWSki9Bw6QIZKMEZzZs5nBaQRnAMkZZxVXDWTULjODTBeoqSMO56PjjPKSjFO0YCArplS0zNNsBAsFC4EGrR7k2CCvMj4+AXMerDtpxZWfRa+lSvGsa+GSePN7Hnh9R0/7+0fY2JTQSsQ89Z5RpqW6AKqAjMCCRlBGHq5GrjHYtBmdIeNolTjIkjok/b5qDRfIDGmM6ZlPypHsOlUFPDLwqV0ngMyRJa63S5LhNNwgC6FEh8ZDWNe+6BN/+rpfnS2jUlLMQ3kIu2v16q8Uupee/6nZOWgoM1EI3LajsZdeFKI7PQ0y3RB2QxKoPoKy0hmVlS9SS0VUD9QsX93YDYjinHSiexssToZygQxTZXeeC1QOcNJapCq13yW4ekaoBjI9+rbTgJGGzY/74WT5IEQlaD214/JPXQfXVbXBdFVXnXzqGxsb/DceHJnkdlfECgt0triDIA0yahFqUJfVKTY4wYqzGrOWYDdeapQGFi2Ojs1KBw5aXCCLwUKddUgBCQMOg84qLyX1HHkmNFGjrnFZNJFk4AYZQXTYS9RxGjY5zvcERvvHS7CokL+24+wPKAcsF1RaihnibPVVnyr0rD7/+lIFCuXSFMtFakY5InFLq2yQwXxAyuLILClHoTaQoU51jbycgCfoGhzMQKGq7WhL47gexAS1DSQA92CJOOoOYEorks5TA1fnETcwsfJ4NSCDJA1TlD6dCpktFkLDqYsuu/6667QU06nwJkxYDOe8rrm98PrJyVmeOuLbojQIcMoUV1O3BIDqQMqSHGClM0S2o8IZDZAqwyk5ZX7z2oekShxY+avFV3Z5qMNc5UWsF6KQb+OjUJqhMDtNYY79lmYpVEpUxBnAs0FAIFWmMvXjZlDISmBlNYMZLPDj92UaNpPk0YOzIfQ1FK8uvvSmDXZ2hpu/ffHr3lqpQGt5djLe8UPAuWkhFpFgdI6ScVJ8xvv/ICEwtdYep3PEAbovVyiMHIzEX7kMVUOJdcTIcASjIxF0dntQKBCDhSm6tfcEgEWLAiauP6673qUERNcDS0q9gSRCecqgRm0CUAxAs5MMSNMxiDwS72AgapsadxDJxVI+67FCkUAd+yvUkVhE0Pn6KWnUmERiiTX0PljPCJgbYMBMyuYro5nKlB9GzT3LWs95K3v0pIwSDnu1/Bk2vPEbKzr7Vl4+M1VhD8RqDE8DhCKAIOJVSXHFaKpiGCy1gszz5HNGwuQUBb4raYb9cu5ubvWgrt5uAjPUFz1oafXF9rh9u0OoZ++3tnnsOREdJ7g9ctBJE6lCIM1EuLF1HH9GiQFU1S5xneJKcn8WkZkTaZSU5yhMjlKYGY+Ab8ZoaiPQudaH1g4PGltjAAX5OGMuveYY+CZHKIwPMuY5EMHB/kgAr5G1SQP7y8mepHY/YaEld4BocNj1g6T+AGZb6DpZz9mfx5zwMMkYf2XbyVe8Y9WXv3bLK58bUMm0BGttP+mSQh2cODk0yZL4jHqxZtsECLg5JMU9mGNowq0qkQZZhLgcYs7lF5MTFIYGQ8FpfUs8WLMmB0uX+XDahjy0tFVfIdK5yINr318PJ633Yc+OEF7ZHMKBvaFgl/ZO1nmNRPicqVzhhiUZ0DTgDEaBNKMkDy0JYN2L2SdGQ1hiABmkMDUWQRMDxtpzcrBiXQB9q3zoXMIAxqRusZlAnkleBUYh5Rggp8coHByIYJgxzr4tFdj1fAX2vhTB/q0VaGDt0sLAGeTiumk8gQ0yq8FqAZliTHCCjEV7dHxuhixtrj9jUctJr2OPbwEDYMw46+7c8LrZGQhCPt3IWsLgVlU4FvsYZHYcxPqeWJWhxOoo2QkcWL5PYG42ggHGnSXWmCtXBvCa1+bhrLPzsHpNAI1NCbAqTGUGgZk5Lz8MKeRyBE5nafjfBJMOW14swzOPl+EPD5cE4CaYZOha5EOOSYeoQg2Q4ZbLkrhgSStDzWJmg0RC8KMQ+MOxIQaugRAaWzw4+wpG44V5WHNmDroYE/lBdcnM1WJjG0D3CsYd5+aEVBvYFcLWJ8rwwn0l2Pp4Bfa9UoF2xmCNDGyCdrxHW9fJVpe6BYEg6WZjUYMMtYP+YZFz4RyUKvWFJZ1nXQAxwEQQVv/5PX+0bOmqN/z5bIn2hOUZ3qpqe6xhWxDAmZrcqahK0hEjDnM2zpOrRP43xrhz374KNDURuPKqOrj+gw3w+jfUQ+9iX3CzbgZRRhq8BNVaxXH7a/GSAE7jID3RZ64bgL0MZENMCuQZwOrqPYN7FSCIg06jHiTd+K73iGAcBnxmPw3uiWCKqfs1Z+TgzR+th6s/XA+r2TVXcZ5XHVyu4DGGbGr3YPn6HJz0OibZGdNMMAD3vxIKm66+gcmVgKTrB6BP8qHWc9FrFrJS9XTUP86SQsGvZyZJkVkn5V9tHXlomkcIgK0881MXdfad/LHp8ekCpaFW1AbIIAtIZqEpIFmSDHeQAtcQa5hB9reGSaqPfawBrntXA3R1+8jOScrHCx3sgONwGl5GV48PZ74mD21dHuzdxbh9ZyTi6hs8bUykGpC4GxhhWXcIcdDB1fIc6+z9uyK+dQcuvqYA7/1sA5zM6MjXkRSdNQdL6haYnbl8fQCrmbqdY/bqHmYWTLEBUR0zB4IcX2Kf7kOiNI5mSlQ5CsYo1yCNIHMCRfL0EQ2gtRAU944N3PfswE07+XMBsNVnffqdLZ1LrpqbnBLjQHl0gwMsaYmUGhlhohySTMUJcLGLA0wljjJ75DXn5uGzf9EMG84oaI6OoiTfBXUASoM7kKvhVWtzsPaUnBgA7NwaivhiY6x+tTFbg7TKlPAyjo/2ZpmL4cDOUEipt3+8Aa75ZFFci9cUqL2F1w3TiNuIq8Z1F+XFpkFun00wSV3HNEKuKsiIWTdUZ3RrFu/oZ5DjneZ8rm5wcvipx/Z9/wn+0F933XX5nrZ3fqipsWXD9MQUSOWYVm3gBpWOc6kSeWXHqWH4IBsRjTFwXcxskf/rc83Q0+NLSuM/71Aa36ZLlhuFCQ0dXWzAcE6e2WQV2M64nXdyPQMZjRx1RTckI85mRC6tuB3JwcVHhe/5NFP376oXRj4PYSjpOuT90mb9eBADJg+ELbeG2WgFpiK3b6zA+ABTl82eADw+zsRUl0mtjPam6brrqLRgYeZdCG31dd5YKdz6wM62uwDup563a01boaHzhDlmevEetZ2RWoWja2qXSKBKOrX6QqwTjTuBSZKDbIh+kIHr/PPy8BefaYY2ydmhPFHMOwyNj4PYCqwMXxa6mDH8p/+jmQEtB8MM6BOMFg8Z2rbXHquGrMl48Zof12FgDzfmCVz3qQa48K11+n0OdG6XvVrGwUGYAb5kIhkuZiPpN/5pEXL1zARh0poaEzionwgkFQOrb5W6BMOU01FKzCXjCB+mmbuiJd+16qy2/ib+zKtrX7q4sal+cWmuZBYCbpCpiEwAUjfIVFqu/maZXTIwUIF1JwXwqf/aCK0IXIdDamUFIuctVEdwl8ZHP90IK9b4MLifGcdsZEbsBUwAC5ri4X98EMGl1RtZJ1/89gRcvJO9Q9qAX1vQIJNEXfKBerjgj+rYqBuE70ycveYQFNTV3kS/Mf9WOPkql2CzbGTeVWxe3Nu2Whw94JUnDvZFZdrF2SoFJEiDxZi3s+J0BzhAFt9zW4AZvWyo3snU1PvfV4S+xbGn5EiDS5MoVZOyXU5YG8DbGRAame9p8EBoggjAmAzXdVR1sqQcz3ea+fCmmHvknNfn4ar3FfULRxpcKgiQIbvsik/UwymX5GCCzWzMTlHNZEa/qEBNARO3BdVgqybJ4nYi8YrVKOqN/LFeQU9Iw0VhRIvExZG4NFccZMRlgIxXbnwiEnNtl19WgAvOi7k7io4OuDS50lZSkumiN9TBeYyeGeYtn5pMOiFJkHC+bmQH4Hg9hhnzLFkVwBvfUy9Gd+r50QCXCp50jfBQZP7Di6+vg/ZlHozsj4zJa9dGE/6QXwsXGpIm6qBkvRnHCiKN7HAmRJrLDFf8LmBWVxsVB+6qeUTQ2yqNeTuQz2giGqs6YgEllJlwR+jQSAhrVgdw5eV1mSPQoxGUFOPlcqftpcz39sLGMgzsjZjrIkaDqoMxN6mH96htZPzkaJzhuZczv9tp8a4+Mbd4FMGlgu8nhv+JzGF9OqPp11+bgQPAQZesFVMOc1VXoqQCSZ4TG5QEUgMeFefnxI6s/NxU2MafsemgsGgPEfjcWWqBYAbyJV0pgy8108/Vx3QkbLDzzi3A6pWyA6LDb9DXGjxUz7WnBHAG64hKGK9i4EGv6wJHHS0zgtdjjKmhpWzK56yL8zqOHKO6Yfo4wDdcWYC+k32mvuWgS53IKob08s9Xf5Kh1Du++5oguxOk6cEHOcSnvudRYR8EfME9JghLKxWwJANLqjlXGeA4SFTKKKtcH/PMbzg9nwyTj7Lkygpciq1nnvXHHphjU0w0mVRXFbAkGQ94ymyOuSW4i/pElsfSVbFdqedhj1EgXkLDitMDeMdfNrC5yzCeSsLag/+HRTIAzEt2xnvdHWy+NfLJc9/262Ava1dKqK9tCwss2hlIzMYEW13MBzJ2wyXDXDmeY1xzwvHRATzg8rnBv5zR99TvS8yOiDtBz0fKd3Db4Dhuu7WwqZtVbOIaT20d8yD7iXv0N1xRYH9wFAKBlc/mPfglV5HUNNldKgEsYzBTJUJafcjyYJZxOJ//W740gObmY6U33EGpSb7aYvEyX6gUvq4sa2BjD164ccu99p29Pixenhhcx4N0Jh44V6Ue8XL9uFTR08YoQgUKbj9JFSBRYqVD+czOgVgq09PtaZvreFGPig4+pdLV44m1WJwhdD0sm9MYWdPYxcJ3xnSytG3dx8Cinycci0HG3JQsm5D4bF7qMNBtsOh4/IuAJH6szaJUdkKJOeAa2XRMW8vxJb3s0NJGoMimWSql+N4erBgOVnkdVqhwrPKFgsXG40g9HgchlmDqzgUycEgrl7rAnSD/wxKAuyjqCiBWlh7Pob7eYzZUbDMqG9RlKgAk9ePGPZcSfGAQHH8C7JiGtDhx2R0U3B79JNrsBAKpzRARjdcwBce3AIvnCX05L4ykFXVJeZBxFPREMznO63e0gwdo6oCised86lLHAbjtNWJKRrFcOOSSAY7rwOmLQmWXUTC8jApU1LxXo8owOjYG9fEcPL2qVnNkFZCpUCMA1TWP5pJhphTB9Mzx3QMzM9ydwje0g1kXCqnRs9agXmwCcAdtpQzHXVDLn45q8NSP2mGD1ZpjhX9q+QpUsdcAUisQuB+G7xI6OH5YvrF0WANu/HE23cO/BsfX7KeZKN0u/NrPESG9eNqZ6eOPgaIj8qG+6qFBDuYCfUyjJfaV5JEzlJlee2xzqf1z2oNPE0dsHXM+jo6FYiWFOOzCI8eFoxUH7vsaOBDCzFQEDd0BJHOv6gttJJmL03FydS6zL4dYWj7Z3d51HBliNNYenFC+M4kv2+HLkjgD2YGgNPrBAoHZ2u4LZhveGc+2B/Ex/JAMxeV1DCSaXCOwaC2KXBTGJDdAaqKce7f5krPtuytiyqi9lRx3frCR4RB276qIBsrlbUZKPgOYJEwMfL7r58B+Cnt2ssn8U+LZt+OBgRR9PGy8Zw5+d8OsWDre0BR3kDY1URpiC2oKbieCMlFRZFtDMxMgEbz0gAZYMkrCNpPYSCtWVRDDuCWWzUWsOOeUEi8oiEH28vYKbN1ehnPPKJh5HqOAJ9tfebkCO7aFwtHqqXMeEVMlUpzqDbcq8BUKg/tCeOXFMlz4hoIwCY55QJ0/Nx3BU3eV4NFfzkFHb8DqCAZwCBIaeGyDxzpYaxHUJliyNAe8bSIY7o9NIfOEYEskapVIqGxMkj3/CFB1bpKHVqaX9+wP4cnnSnDOhoKx7OVYBAxuvkbt2afLMMBUXG+vbBZiMhW3TbUUV0FexxtlCWx6pgLbGVDXrMvFewGO4WoR4RqSZe94vgK7t1Sgh821di/z02CBNKhScS6QWXHNBSKYr26aFXxQ2PrJGRSuUSDIA3ZUXMrJStOjK1cc/20oxgB9+A9z8PK2eLjlkWQF5tEOaj0YD5s3leGpP5SY5PHEmQ9ifRyuEzYj1D0KXA3xye6dr1TgDw/NOZc3Hc2AmSdiI9yN95bgwM4IWnt8sQGES2g+UvbUte+4DpR9mdwTRzqC3uV5kyABduLJR+pMP8NxQAywGIDDIIM0GNU974QO1gmbt1XgjntnDGAdbZBhycKN3rt/Mws7d1TEhLcY+EgxhdfDmdNG8oMMoE00seyaz7s9fE8JXn5BMpB39OumNpcogL34WBmevLsEPjPs6xvBnHu2BYYV51rEgK/VyzpPi5m08DZ8XQ5JFZ81ZVt/kD2lRCG1wYCL7OYmT0iJO++bgwcem42JkDP+R4vTo8jk8PvunYUHH5gT00TFBk8Pduz5RxGIIw5A+8M62GT+ru0h3P6zabHplgffT3ZLHfFA4/r5UsvzMzDu+fEMDO6NoK3HN/pf9w1JC5RqAASUzpDkwmlvfvDHi6yCxC9NpJRRkLRDnAXZBKIOMZDNiOjp9mFwOILv/nwKdu+LRxu+n3T8kQwKyL6cM3zl5TL8/CfTMDFJxe5vo35GHdOmgnqiSOajz4ZmT5w98eh9Jbj1J9M6vecdBZDRmAYf2Xx3fHsann2gDM1Mc9Q1gGEKySRxsPsQHABE5oGuuy2IiNmHnp2r4Zm37apUo6fBCWATAQbXcxDVMxtnEfMVPftSGf7h38bFzm4eFMiOlEpRAFbg4gb9N742AdteCaFbHIiiNi44OFmP6dOSGp+8zQHKt8NxFXvrT2fhnttm4tQk7ngOsiPBRGrZkI++HnTXD6fh3p/NMdXoiVN7wgiyVR5A5lwrqHdTNqhpRoCVlodkPVgVILkSZ4lam6jUMhd2X2G92NrmM1+YDw8+XoK//eoYjIwmIOPhcM7rqcYXG1QlSx1gTtG///IYPPVUGToY2JuY5Anl7m+nuqCOOmNJLblZ7SDqXurDJPP33fCNafjdr2Z0e2hJfRiZiMr8fOQTuPsnM/DLf52BuTmAziXSrkSzNqoOtJokc9hZdvsk9qh6Row8EpKQenT5utQS4bhRqXRDENNFIYkwdhuR5Fo7X7mUYk+7u3xxfe8jczA7Nwp//rFmWLk80CBQEudQh/lKHRKSAJeHl5jk/Nd/nYCNT1fYoIMBvdMXc4kGzRS1k10HJOXMtomdsRyo3G3Rw0DWvyuE731lWoDtTe+qF05OseNHGeKvYrserh8esPzqO9Pw6x/MAt+tz2ngZ4ZFoVUPMOsHKC/bl4mFVrZ7KvmeJmZEDTBD7KN7fPyjbEPNufOdH6bBStMgiyTyFy3yReM8wlwEIwdH4YPXNcDrX1cnNmGoRsOSrJbOMGwABE7u67rrt7Pwc2aA79gRQheTXB1tElwUjOkf3Oig6h+lO0CDDLTCEFd8EWKBOTMXr/BhPwPZj745DXuZl//aDxShe3HshyLoyFzcYdUrl5SltsupsJeNzm9nNtdjv45XS/YwZuVTQpwW3Xa20AB32aqvVd+n+jBKJBrB/lEw1U48F4kzT0krkwgT+TEFBprBklaQJlBFKJBxScY5/iXmRf/yNyZh4wsleMvl9XDyGu6sXPiUkv0+P7DuReZhv5O5Iu6/f05MZi9mndzUxO0SSZjVmIDqgOPmlWQojm9+5YsX+1YEMNgfwm9umYVdzE92+Vvr4NyL6qCpheg2VyQr35yrzkLV88P3rFmCMTbF9TibBnrg5jnY9kKFuUs86Oz1hI8xDCmgJk/NIYOjDqoNqSsdoLaxMSLbSfStfC/AJ65QSGcOSHIZRGjOtkDmIMKumEa+BBk3jFuYD6m+EMCBwRBuZDbLxufLcMG5BTh3Qx5OXJ2H5gUuReZ5jo1HsJl5r594sgSPPVaCnfwoJebsXcE6nJ9pyletqjri3CXu3ZuIIQ0kHB3/UhlHBMg8Nn2yaIkPEwdjT/+u7VPixMWzz8/DyafnoLvPN05szJo+E6pevseZZmBvCJseL8NG5mJ56ckKzLJBaydnnFaiVxETmRALDVvlA80GoC3lDAmPKo5miwyBFSg3BSVpsYjtkJTIRM/5/7wxqQScmKejYIKTgHtqSMaJCWZmKyxZHMDkVATbdofsbwruZ/bZyWtzYqvbir4AFjEXB1/Xz0eiwmss6aww9TfD/E4H2WCBr9jYzdK/vLUiALaPzRFy24dLLT6bwMsMIzCBoZhBMY4VpzsZ7HQ02VaPGQqSRhdSiT1tbudr9gM4yFw0jzLP+gtsgHHC6gCWrQpg5Yk+rD+nIEag1QKv79QEcwz/xww8fOcsDO+LYIbd8/Ne+1Z7hs8tkUhUaAKKCLfBogCI+wnnMd8Bzgq4dgjsz2E5QUbNlqUuAkFyLkDKXstKZ6AepFhlkY0NbLa/GMDUNIV9AxHs2DsLD/yeQBfz5XS1eeKop6ZG5tfJx3YaBws/35UfHsyPhOInJo4MU7F4sI67RJidV6yPT5kOKaQmdlNc6ABLyibTcSTFvckiWIryIfEIMwCxc6mN1YWvH3ueDTR+z8DGn33kzwEue0s9zBeGmLp98PZZePL+Eqw+NceA5Yu9ALz1o4ikO12L5JgOu391W6C+MLQRlniQBp18JblAWixQjULtzFxSx0aoATKKVKCcFAZiSIAsLrBFtPJFNTCgNRZ9sUqUr4QdHIlg7/5IvKBOSCSSYOEgDWN7jZ/N2sAA2NlJhFRUwIiiKvWrAWQYSBhwzpEXRa9Z+ShPO1831toO0L+bH6tAxY7yWgLXEs0dnjDiW9nUljhnQ7a9Gs2JciVtZj85MlRAUXXIEiLUTK7bA9VXaUMV5Cf6UAYEnEBSRjrm3riy2RLJLeUy4jDIcMU5kTm+DIRAS5Mvv4YRf7BANyrE56FyRyY/WkHvu+Q8HZFEJQAYdojK31bd2RIJnMN0VxzMk04YxPJwkgZmX+YCTx/lOV/gixvzdXG7mG1PDJeBak9Q1bWYwa6DphsXlsrEJcVB96GusqxKgBvAAIHNvZICg1tRZ6UNX+I2DBHSDUCpGwrGqATXT5Tv8SOBiKCcyIf2vgwjHUlLK0x3avACdj0gxb0uRnGlo6hOhCSGf5wuzgVP3SxktKwlBU5DMH0SZNSK00uOyLyrk6lVpqqfzg4zqVW+CrEnHzcIujcSIAACQMqrm4qTd/oZmGVgsWznicvE+duNSeUFRT6SdDq56oGa9QCbLrDyR7TadTBodqQzXkYMqO/BMd+5AHDZNCqmSWghmvgUnSTdF9XiUu9B8h51xFGrHp5KACgxRmYKZOAAIDULTgqikMof3JznrBi4iXeDmmrRUTs4wRBLqXSIkKoMhe/lu9RuQ017Is51OtRhNQUXLTZjyvLBCUAFcZrUD8Xpatu4AEe7QVow4XTxagpXY6EMs5CuSSTq+DqTCGpxK3UQbzyzCaSQzXXUBiAx06VooenycJ4YEK44ACdws9oNHCCz88blLShYDH4otOisXOAEMx7H4eJTAET3EV5wmEngPAUlxEtxTCGDC0yQpVQyIjA1OY7zQnE6rVHp5Mu4blpoik67PC2twNGQ1A0y42U7LgtkQuImcfgDorUEzeAuwGfQ6QJ8LFHpgvs+qw91Ouln1DZYCtmOgqAq0knyzJkuOcZcNayTeAJVAWjTkqJLpSNUA4kCLo8mjZsBakNyugBo1XFeezSDOTXIrHrMG0j6fj4gZcUledBscKJg9BN+BmY6JblSNhitAjLFNSkuoLigBEg2AbbdUQ1kWCK4aHGmA0enU1Pi4XSAysvk4GpgAag6QKKO8rTk0W1GzXQLDDSD5hSdWXHVAIjpdUhHwOnAvDdUpOqArAaZT+xjyaVKTnNrkqMGYBWQZYIHVdRIR8ES38QARKISJA1EKwgzXVbZVnlg1x3ArdbBSldFOtaMMZrOMyUMHP2ZirPTKfe86l/cfuo9uy9oUicK6ToYNlgW16XqRxwNDYgoBDKzoknxCZCo2WCuxkJlOCUgft9qFPXd6lQejjzNdJBmogxmwHlmx5m7lExQ0artbQcKDoaHdOeCo48MCWtLK4J+iZmj7TnQsXg1joNRjCXTFDKIQgTR+QgGBzgzgBvnS9LqK8VZVUCGnhtxumzi7PQkTwouULtUyXwgc0njRMISIw6XkQLGfAEDFSxJRkwAYgnvshXBApUp4VH9vGp9CCkprkLqCE1NNLUaCBFlZ2aAzNXIrnQyxgnqDCC5OLSWSifTDpCSgKk6Wnk6JRkuh6bztOmZLx3u2FqDTovLsxicVonT+RBH3jodMe51ueBob5wO5eXp5WDVGsTRwSn7AYPMSgfV4iTIaEa6rE6nVeJ0kfqeuO0O1UF6BJUeoNhgzFLVug6o08GiBew8wQT4QoOmoQqoswDvlHIonVrmk1LrkNG/OF1KgoGVAU5gE2FnBmCKVxcR8wIwcdTa6VwgS0k1K05FGGDjEa506n1Ctf3ozNMBMoMW6qANveesA5j5LDgQSKk9msEArrjkBsC5LU+DjBpliqeovkbf474GdcIhChgQ+r0sIFWpGE6XvJuUnAKSg/j5QKYysjs9pRIyOtadp9nQKefuAgGfBTLqoMUA3DyBEnAa3irOKM+uszOOuCWZI86QalZ5tg0mjm/C0im1XIckElARn5qFz4jDDabX5dN4daXqKOLKE9Ei4kDjLN6RxJfrUEiWvEC8ayj+EorcLKITgMFZUkglTKNopZgWVGPibhcjHZ0/T6DJK+ZiPsx2FlKyAmJIA+QU5YCYwbnZg8q62n0IaTqNZ6g9nXHyQbImX2ZrSAuSTZRCPcmIc1ZMxhG0rQnH6c4TqpLoe/kISgxQ/OTA6VkqtmWJT+PJ9V9K6Ajg8YWI7EEhR6Chnohl1TmfGEtqXAsEjUaCBGTGtjwjzvEMx6E2dIEaIGEkvr5taiqCsTLA5GRtmyX5xg/+6ZpyhZpLfDCoAbWNqz/BBBKmk6aYAa9UJm4AIrCjasabPgwuMOk1GwsgLcmo1cg4rXFhxikxTFCcnG2KT9xhDyamKIyMReLkwQYGlu42D7o6fOho9qC5gYiVq1xqcYnGP5wg3h+NYHA4hKERCgMjkVhW3c7e52vxPbn51LX9zJZ48Y8FMuoAGWIE3A7g6nSS+IX4FrrxMSq+tDs5Holl3c3NRsrswPIbPUhh764QGps8aG2Lv8utOtuQnJAhrWxJBqh/nSBz1McVx+8Rn+g1+fjwX7AaD8CBdHBwMOZWcHOFzck4TmxCZTdjk8DAEYrKL+/14ZTVOThpRQArFgfQ2xVv+igW4g9YgaxQOeQH+EYCkP2DIezcXYEt2yvw4stl2Nsfxrut231oapASDe1ydql14ugAsDrBqAeAW+Wj/DjAuQQeHorERlz+pd/Tz8zDihN8WL02gA1nFaCW0MXa5NKr6+Lvnu+PYPeOSOxMb+vwIC832dp9YdAHVhy1BAYBJ5BS/ZkhqbHdmSyZ5hkp+4hkcCsqSDWaQbyKo5DqEefSatkJnlzqPMcav38wEp+dOXFZABeflYfzN+Th5BNyDBhe1RWf/NOmTQ0+dHf6cNIq1sqvBbFt7QUGsMeeLsHDT8zB9l3xtrWeLtYRgSckH8ZIFkcmAIxbFIPQNgd0Xiid2lU+wiQqBxc/YejCS/Jw7mvzcOppOehd7OvtaDRK2iQVZH58v8Hb3lMPr70oD889WRZb4F54ugy7GUN1djNJyI8njdRuJofEpWnhoOPmBVKy1c+1gUQfHYCXTFPVWsQCGVIlQN2i1mV36GADwsEV6pOFIxMMXAMV6Gjx4ZrL6uAdl9TDulWBsLUONbQwrj7/LL63sgCXnl+A2+6agXsenoNtTLot7g6gudEEmbGLnULaHACr7mCCDFA7KXuU7xXgexgP7A/FMeenrM/B1W+qg/MuKDg/CqbzdAUSb+blx5Hyzbd9ywPxd96lBXjkd3Nw922zsPWFCkxPMXOCHzSHPhQ/L8hwnaCKlKsWp9QYYrTkjFZp6MfvIJBBhqi1QQZmQ9gSMDH0LZXIfvczrh5iqu3UlTn447c3wJXn10HgJzm6JOZ8AafhW7o2rMvDujU5OG3dLNxw0xRs21mBbmbPdbb5YnBAsZow6KRmvmBJY3DUXT7jUomr7f69IVNdBN7ytjp493sboKfXd9LJw3zncARBkk6laWG26VXX1sP6s3Jw0w+m4eG7SrCHSevePl+UqzbgpgRFBsgAS2aw+lfFUTA2lxj15wmlCWKeTYEyUxnoFgNTWgGWbGASrOjA7yGwa2Oe/+0dCpkqo3DxGQX47x9ugtVLkvNYIgXEBQBL11OmwR3IG/ttV9TDambP/fO/T4izYvmeykUdEmQAmaOrzBEidcfxjb78C797GbhaWwj8EQPWO66NDz9RdPG/Qz3YBY9CiSxzCavXJ/+qCXr6puHWH8/Avl0V6F0aQCEvTxcSCUFJlEyQGQIE9yGOAxVn7v1U3ghzRat8aDo4rVWOgOOSsnE8tUSmdgZSMAw/ZfD2D0cwylTjFa+tg//531o0uHga3vEeOTRw4aAaHx8HdcraHPzNZ1vgPKY+hw6GMMj+uCpTHWDUi5j1s5deKybTbh6It5XNzjFw7QuhjY3w/uQTjfDOdxU1uNTO68NxOLDKQ+XJv8n57o82wPs+0QD1jQT27akwd4Y8RRLTrNpVMSKgX6TyKUn3rzMOZ4L6zDyjlaRfMubmqFmQ8Z4BThTnIJDbVUPMAB8eD+FSZsj/zZ80M9srbqlQGrn+YWh8HHx5TKc63I6fsvj5TzeLsy+4W2N0IhLAUHSnGhpQnbD+IKjhJeNwm2vfPjayayLwsY81wpVX1us2UofEvVrGwSGWmOYJildfVw/XMaDxD2AcYEDHUtegWdcJDGmlbmkVANo4oMTYSiJCckYrypxSs2C8QDAL6SkJCBZwVYH84/CzzC5hqnHD2jx8/iMtwq/FA/8Qlfo++ZEInjyLS3XEIjbi/MzHm8S5F/1DFfHJZ2Ixh1EPQPUT9ZXNaQFwkNWNeBSuu64IV12VgEvs6D6Cn/vTIJPEvOXd9XDltXXCITs8EMVn/5MapBUxQQIOyYWFTVJ3AmD1nVhNUY7izuWnzfDTB/mRRuLa/uNx8lq8g+5d6XCcuucfJt3N/FTc7vnkNQ2wbFHc4mF0+DnbFZR6VpLsRDawuP66BnHWxX4ODBtIKJhSXAKLJpPjPF9+1us4+7vwwgK889pi/O5RAJcK6gRFQRKj6doPFuGs8/MwdjASxxPozw1iaUXTTOQyFYw4B8jU+bYYnMEwa4zyARZRYnOSMyHovKilpmkyWiAoLv0e6NGInQ8fWfCPYZUZ0D58dRHecHZd/JiqcyaOTiC6QeLrKy6sE2eS3XLnDExMe8JXVs0Ra0yZ6UxjJhlgzLNiuQ/XXlMUB6+I+h0lcKng+YkqbmA+tzcxdbljawWGmBRbwr8pTjS+TMkMMO+0ER40VWsLFYLVbMh87pkeVMoE5sYLBnCMdwXAqHyWyEzjfZKASqdF7/BO2z9cgRU9Abzr0qJRs6P9IU/1AQjeUHya5epL6+Hp58uwj3nGi0U/oYvMB7L4+CZex3E2p0iYHXfJRQU45eT4e0VHG1wqKJuT03sqc1+89pIC3P7TGZhiAoV/MlpUz54yk4MVPdOB7Las6aY4I0jZ2yoEV51eB//93YTpaf41tOYUoQuVKjbgAN1zmrm3vrvVF/ODPNjHbh/N4KFOWH8S64QzC/CzW6dgeoZNRUnpo6phuGFQveI4vrqDwAhTQ6uYq+B15yVTPkda5WcGXq4ECffHncO8/k8/WoLB/gjqJQNp8JDEHMDSCudFIS3JdLsAAB5JU7QgLFjbl4dG7h1gf211ARzNwI1P7FA9loHPa57Fpm3u/70PY2xEWazzdYvaIBNBxckO4m4JzvWnnZqDE1Yk7hZyDKtHEAOtPimAE9cHbIJ8jk20x4f9YQ9AsqIFEkkGkL3ECdzgVEHNd3sHpw7jedoLDLng2H/SD5d/4spASCB+UqJylxgjKPueJPeT0/xodg/WMh8bX+VxvIX6ogcrGcj4V+G4A5gH7Wog4HYzyaCNfWyvWenwuSDKXcODFx07fB03QTUSP0FxOZte4UYyd1mIOP6fZQirCxXHByn8NMWeHpZ+aWJwHWvmAYsGbuDzIzr52jrDVrLsJ5ePzAWyLPcUDxE+OuD/9KA6IZfzhAOWH7fJbUV7mK79RsTshFC6aPhKjq6uY2DR1xg6mEuIf3SiUpGHL1cDElSJc0hx9YUi7L7g4T8BZgW+1owf3Vnin1Ai6YbWXIriuA+Rjx5bWuOVtMdr4MebN7fxLqdiVQYPTo++vMZ1p8hey0qn3xdLZOLr/wSYFbj0KuTjJTbUGgkbkgxdKydxPUsbHN1x0oICX4xYxxnAT87B5SFLUoNtj1n31aScMvP/E2BW4BPSwtOPntH5QCaHWirt8RqI/LgoGizKiGxJrd+jZt0NiUezQMbKq8/BUQ940vl4C2JKSzp+qcuAdXA+/5RyPJFNDclwvIWIyq+98RvLsDdsTHTvAhxYo0nX5LiSXd6x6GdCAI7XfuDHpfMNJGL5jqMT9D1SF/FZ/VSkrZThuA3lEggXTCSXQrnsLsiyrWwpB25JJn5QnsG9m2ZI74sApVk2UXtwyijLDgTnDkmmNo1GGvn+9Fw8zrjglAKsPyF3zLz3roAdonyudIoBhX9ESnu0kfORB1xHHsVVI7fDRnla5gIoFo9PQ5/vXuIffxBOVL1hEfSFPW2U7kxIO17le6lVzTIE971UhhduDIGWQwinJzRgCC7Yuk7HEeM9ItcfE0kJJ3hsklWOOfg+d20jnLIiF3MQ6tjjIZTEppMQJrlPqyPmAAUypT7whC/IOPFxdPb6gaEIBoZD6Oo8frgHtzGf7B4ajMTUkedDesshBpm8Bwk4jTcJLB1npUsb+WKwFH/MXP+xNyOa3Itr6o4Tz2lSGbyzRK9mZT98fyKXYpv3hHBQbTA9TsClOoCvbN2xtwIV/t2kPEmrC8W1lirhn+Wrq/egf3+8XU4FejzYAYiG3bsqMHSAz0XKithnTlg2J3bJiGfUfFe9h9MlqjTOPGjMQbioheGt5EHZ9+IGs9UeNcoR4NRry+w4Q5KpuJhavndx2/4KbNpZhgtPjVduqKU6xyoIe0QKnM3bKvDyzlBs8lVLjAX4HN/M1OpS1pGvXuUA4/sw33BJnZgGO9YBu1mm2ZTg1pcqMMGYu48vTceqjaK6AqTmJhWDYammJBsAuEeTXryfyWNvlXQsJCITM5+9bFYhFo8a9Aw6NgZ1Oo4iKlZQ7BgI4ZEX546L0RZWH3yn9RPPl8RK25YW61PmVkMbzlbZFnwtPB95bnyhDFu3xVKM530sR8u4fi9vrsCLz5chx+ZJgwKgETKtugK52vyja40+D3L/jMCVB15uGkw8mSCzUIqJoHbBKJsUEexBI3Py8fb+3TMl2LQrxrX4Atox6gS1HoyHZ7eU4OGNc8xRGn8qkDrqp4N9LaVbW6sPr+yswIOPzRrq8VioSgwu7j557OE58YnD1jb9ISfUh2mQuQSMvs4AmVarrJsJDTiuuASLRnkf2wYRwQmw6APzOkkQ56zOXdf2CiKCv8LXgj3HVORND81oKaZU5dEMWDVy4/62+2fhFWYf8glvvCchJY2tOqlr/px/NJVH3/3QHDy/WTKQd/SlmPIzKoA981QJHnlgTtiVxUbi2HMR/29oHEvA1DrJLUcCs7QSjfIrj0beAUbJJKXpVsAgS6lE+QIFSBOM9DvOjIOolVWQ70+8+dFZuOvJWRHleaCPYzoaIbJsjjsfnoG7Hplj84ie+FalalgnyFSdAVJSjqtIfjgL39T7s19OC38aD/aOnyMd8CrayYkIbv2PGbHLqbPLA3uAkh5oUV13ShzSCxAOXHFUjKoncvn8AX7v5Zs693p5MuwCmJEYIdapLnWHxPrC6BxNcDz27GOd0M/8MV+5bRK274/tlcAz9y4eqaCArPxwL24vw3cYGMamqNhlJD0shgR2STJA0YpmPgPQ2OiJsyfuZYD98c2JX1E4Y48CyNRa/JguCj+9YUp8TrqVSea6oifAl2piS1VS7Dsijvcz4sQHUVmZAfj7vbBjH3/mjQ1v658bmdwb8O2/aTgnAwQL+SmjzUa3ZRSr93kbcxunryOAp14pw/97wxgMT8Tg5lv8Q3rkVEoUmeDaN1iBv/vOOGzeUYHebp8Z6uhTyxlMpeMAPccAhPgj99x985NfzMBtd03Hr5FkM8aRYCK851KFW/5jGm6/dVas1u1gNEWUOgZgmHZVJ3WkKU0LCpvBMMiYCM/lPebznNgzs3/7AH/kLZsYG65Mj2wPuGqgbhbDIKPooVKPYBFhJLKueTTfxtbBhvUdzT7cxQz+z317FAbHJMjk6YRhVIVzFhhU44sNqhJcewcq8PlvjMMjz5XEsU4tTPKEcg27YXdgWwPn6VAz/DqKVQQs7vWFc/nr35+CX909o5tBbSs7nDanZhwErl8y6XnD95maZlZIz2JfUBdZ9aAOyYz7Vz3AggK/q9/X+UTiWK3J2f7tj418dYI/8fv7n4y6TnnnumLvsteXJ+diuUrSokw9UNKTyP8ImPdAtb2PvPTyMBUAvRuH3zUxkc3XUj3LJMhLu8uwnnn4OxnoBLcrtUvhkL39Kj1IYOkR49YS/PU3x+FRBu4uBq5ufgBKCIh4iz+I+ateMzSJekd2SD7gS388OMDcMs9sKgsVuW5tTuxqV4OLV3P2Bq6f2uvJAx+w/JipxR/fMMPmRoH5vJhkLkgGQ22v64ju9TXuQ9QS8/Q99f08Kebz0c7+P/zoue03PM4fC8y3nvyW1sZlJ70lnKnkaRSCC2A6I/ueZHSGkSZGHiH4Lv5tro+du88zL/OTL5fYPYHVi+Njm1QnYslYS2dgFeQhYPFNJjffOwP/84cTDGQVsfl3EbNNzAEGSTWk+FGdY90bDY3S8Ty51dHImIgfTbCR+dj4ju+VKwKxyVcxkb3XcCF1U+2j0u1kg4tvfXMSbmNqMQxJDK68PMIJgacayGwmSuLMfbF23fldLqhnAzh/eMueJ/5l656bdvKHAmC5JevnmlZccEXey/VU5qbm34nhaGTAly6ki7lJU0KotmouxiPLrWyk8/sXmbNzJGQq1INFbZ4AJUGNKdRQlO39D9EBbqoafAnOUwy837hpEr53+xQcGKGwdJEP7S1efBqgRacBJLs+YAIhq4MUyPgiPw4ovink2RfL8OLmsojrZeWrzSEYJHhXth04uHhdfN/sohHWXnfdNQvf/e4kM+jLbLLdg8V9QXI+WAosaUHgBFmqfiQrjo0RI9JUbIIyhE9s3HPz1/r77xfGpwDYZN+FU92LTj2z2NZ4ZmmSPyc0S4qZuWcQj+LTIDPfFXYca9SGggetDT6MMLvlCQaGp9kAYO9wvJW/pdGHulwCTOzDSpFFEvXKN8I+wWYNfvrbafj2rdPwIFOJ/GTDZcw+KhbiFRA0E0gkUyWociBuKQOQRhzEIOODl+ZGXxyuwk9Z5Bt8tzGzYHomEgMe7hpRB+1Vk2Zx3WK6+IpbfjTUfQ/Mws9vnIFf/WoWDrB5xm42WOmUk+3YF1YNZGYdabruUKVd4gs28CSkubEeBkb6b/rl3W/7hZmGhQ1/fM+HOs+77FsTu0cCiPh07/x7rQUp1Mwo694VJ/15Mi4+0Jd3OD/b4cBIzHorewI4jdlmXG2+5qQ8nL0mV9Neyt9vmoPv3zENzzB7a/BgJD4k38Psrcb6eE06lbsejA7AdCpJpu9Bi12DkVLpADGU+S4nmx+lNDQSwcR4BK1soLN2Favb8oD9BnDOGQU2Ap1/JQY/jfqXt8/Ab+9moNofsXnG+LzXdibxOZjVMZz2EQ7q2r6Han1I475RxqUGGaofM9tpENSR3tbi9BNP3/7+Gx96yy2KVj3uqFt2xmjL0te8EUihK+Jqkk8eziPFROa4E4iDK1BcSsopImWcMlrjk6GZ/cCG1wcYOJ7eVoY7H5+FEpsvPG9dnZhymi/cyGytrzKVOMUM3RW9gbC3eH6RBoE8CkGUT9wSF8zKmHHy8XxSjiZxwglJuMokwlc2x8ZUO9nswSNPMM8/Mw2WMbW2dvX8S4x37a7At787BQ8/ykbATNX29vgiT1UGDq5+0fRadVDPNegIOEwFYufJ4BWR5qYWqJTok8/tvPnvdu2/Z1q9rfd/7PjFZ3ZO7n7lrkKLL4QXhdq8BApkhu9LBnt4CwDuiVVqxsV2CGWjTALLWQOewKQYr83wBIXpufmdZAKkTAUuZemWssbnxwCI5cIWPcZXehU9Bp00kVwkPaSXb+gynXFWnsqG5JKmsyPeR9nGJA9nhLGJ2hyAPMs2NjhZsiSADpaHcn2YLgNUV4oSWrRgNwSOA2LGJXlSOy7y/JxwT+wZfO6uB5/+H4OYVg8VT8cHHr0tV4AJL98ANeJL062nlAA1sk2g0QnU2QiaGP0v3gEuzshnf/kajxrgH2Mo5EAvhLNDAhYiS0ueG0Ai1gdwbCBRMKaQjDhUPztOAA0ELzMbzBOLFPksQC3Bz8XfCAhyYM48YNpR+Yq2TAA64qqBTBw0F6tMZtuHXl1dkTPwwO79j9+KyBSvK4CJYsL9Tzw8OVL5TaGVASyqeIz4mn3qCmQq61SlHQ1tOGkJvqZmxgCH5P02pCnOx8rTAJndUZIZqFSrrpUFqfwB3I5YG5xgMmRtrAP65RQzoDjIiHOCTEa4QJYFQH4dCcVASXNDDkbGp2+767E/flLGak7xMElb7/zq3MgL992QL8Jcrq4R4pV2CxBlNlEOSZaExGZISTndIUR2AF1g60NKrWl6CGSo5zTIDGlFzGkTZ+dlSDJcnnMFguN+vqBXrTjyMJjIijOeIQaHKkAyACjfoeKA6ZAUCs189mVs9777fwjpIjTAIn099tCdMyOVOwtdzdyJIj++UlvQdGKiEPH6JVxpXTFq2UcAybBsgR1AbIKsMsGUqol6NkFmxBFIblwAtMvOkmTE3emu+/lCJuAxE0EVKWfVDwNI/LoAqK9pGEWR19FWgNHJ2V/c+Nur75dv++AAmCZn041/XTqw6d5/oXUwky+2sjcjsmApJnPThr+joqlKK0Ra4MySFrXQQO3Go+YLBp3qWrakC2RxfJWP2lOoAk6z0wFM2hbUwGCBAUsugOrq0kpHLWmlIkUenkknKpv5b0O/vtDGd7IPv7zvd/8CRnQSMMD4IEu4LbZ8+4p7pl8e+kGxrcgKFdPDFVhAIGByLCYwczEfURtJINXpKh2+rzW4AFp9lIQAbtGi83KBEzJAhvJVediqe6F1MuhEEtduU8Vktuqc1z40wWRP9jOvRBgFbN6xoyMHuw70f+umO970B/k63wxnaLzMYcvgzju+VGmGncX6djYELnHnzCGBjGYQ7wSZjHF1um1v1BKoJQ3ToIZskDniTKZJL2UBxPV2nAF0K0/xdyg73XT+yWepwap/lrlQi32YiovL48c6B+2trTBTgk2PPvfr/4VKSJlTdrW4FBOevm0//OCugT27Ph8s8Wku38I0btlzZVAtEHBwBEm/ZIJMyz9dSfWZm5pBZtsOGCzgAHwGyAxSnHXIWF9lMxHqZFsCusiuNdAMOjXgwQK4FWfX3egLRxlUnMtX8ooN7XwSvfL81pf/8umnP6L8Xlz7zQswHrSqfP4vl98wWyl/r7G3kWmwnEejcggLtMcIrpgi2lY5qTiSgErnAkAPRT8SC0hSymRKMkjHYellA0qVkVJ7OK8apONC0WW3p/3NbZohuezybVrwe1YdIkrLNPDrve6OAgwMlf/5F3esvV2+wvHiXEzoAliEsqdjd/37X1Sm6HP1XV3soc+/Rlg+HCBTJSihlY6TNSduIM5boEyTqZ7BATKcFqeTLzuZAX/dgkBVj34KZK64WgOxJBJJ5GkmM9hpMQNmpJNtyKZuo5B4QdDa2gKjY+FD9z/1z19IckuS2yFL83N7S6jK5378Xw7u2/Tg+/KdcLCus5sNH8p5llfl1YIMi+xUxcCUZIdqf2WCBSANNvSuU61bas5Mh+jz3ODEaXH5+sECbDCbGU0VXB1kuL11vNX2Fp0cX+UoKufa2jqgrg72PbPpt9c/++zn1IYDbthn7jaoVi0Nspf+7eLnDr6064N1baRS376Ygyx3KCBTFTXEO6QrZnZQIuoWKsk0MEiS0DWCSr2fSUv6UDojDpK0YEnOrLX9NpPVGhCMHMBFrZQhyahFp8s+5JcslJndlW9r7YWGBph++ZUt7/71fVftkKnmHfx589SBq0thjz3xN8tvG31p55/UdXm0rqP3kEBGUEV151nSwgZEAqz4/wX0ga6EweWQAWpEG45LSzKrU2lGng4JaLyDASfru1BuVe1ig9MFeFvtWUACgxaI1aIAF2XgaukFvlhiyysvXf/DG098SL6tJFdVsucTzHKxbfzeY19a8Z3Rl3b9WZFNzNZ19AiQcfG5UJApdWnYSlZFjY4+pECzpaPZkGmJ5EinbwgxJI6hnkjCCikAWsCtKk1qqh2kDPGs8qoCyWpnaZvxJTgVLrlameRqaoRwy7bNH/7hz06+Wb6mvPXzehVq0fwV+Z4g5fEvLf/qyIu7/0ux3Y/q23pYbClPuSRboAtDBKsTjMsU4EjaWM0KNN2RtkTKtEMouNWFxQxOCZBVfhbIbLpqrZ8uRiwlVTeO8oSSM1LUYI9GTEpXKJRzrVxyNUFp646Xr//RT066Qb4tv9kGNe3yrNW05AAKFClP/P/LvjG8Zff76hf50w1dSzhpOSbOooWuvlAXmHvEL9idsMCWl7ZbnEeS1gaEBgtkSSRHOqzaaDqOuiSZVSbOU1zTBQmvVDDcKbZ0lOtl1DmykA14/koljCq8uXNd3b3AfKmjL7+85R03/GTtT2R2StDU7HRfiP+YHw4ZqDR/+NKyn+7fu+tKWAZ7mtoXge/lArGvki7MLkt1ukt1LCRDO39K3cN0mwbqViWZkgx1lgucdnkAboDbnV5TnXQ+SSJbImkmBQl4YtJpLJUSbggutSp+Llfn89Eii96y8bm9l/z45yfeId/kkmvB04YLnaAoyzTC8H/mr5Y/tOvhu84vE/qbhlVsCFvf6jE3GT98KqxVmml7DCB7KYv1W1NwLGVRje4CUhYIwAUknGcVSaYggKUV4GsXOGsMxsAhQwXj8lymAo01Zggkirgbor6+nSzua2YdF/7Hww/fcuEttyx5RuagltYvCFw8HMoMmCpEnPL5yv+6cve+X//Z20YGJj9f6KmbKHb0gVehAZtV4B6hmmwzDTICMN/ZU7Vqy9ifRd0dCwhANqhpuoPsAYmRnmTTmvjIaPKunadLctYSKGozWwJm5GlJ44i1TxiGFd/3Pb93cQ+0dwWj+wdHP/P0k1987yOPXDMgk+Vk0kM6WeNQAAayMA4cfqCFWKj45Geb/nbPxgfeWG6DexpOWwT1TV3cdgpoWPHkmQRVgYZBZtg5kOK6+QPqfZpSF+g12z6ygJSKUxEkVYx+4JJy4rcWCbiQUAXUcZ4Zk/GERrw/oqjiEeaab23vgrbOdjo9RW/f+MxvL/23r7X90333/bUSIsrPdcinhRwqwEAWWpJEiHxe+srFj4zf8/dvm9gz8F+LDf7m5iVdUGAVYBXzqVi8KNJEkGFSKZClRmwACzOEMRgdgEiNoEhGHDjSqXxtNW5LuZQEdEuyQ5HOKbpwnhpIBDOqODKMS62IMoZn/dHY3AU9fW1QqCPPbN2y+6OPPvSta+74xRUbZQreV1xDleEQsI/D4fjwCSfCl3lVnv0hm0L44ee+fsZbv/2L4OSLP9K4ZtX1hYbutZWxEOamxr3QK4sFH4Rvi4sbw2hWLcmoOdIE1/V8QRrC/B+RDa7OcuBlGKcnA+gbI05JVBwHSZz4RWpWX0O6TvZpzWpgZxvltQTquNf0xOVJQRYvG2DtTQI2ldzU1s43jNDSDDy/a/srP9i77Y7vP/jgf8M7gdSargXbW65wuL6sE6L8hHf36Vs/ug9uhb9dfc23vtd06pXvblq59G3FoO0sOgrFcimEysQ4swDK8cy67skkYJC57In5QpyG6nw1yKwyUp9KRultIAG4waJBpqRYBKnBgZlOOV5Mp+2CA3GrbolcsV/W83LQWGyGev4Z5RBmJqbCR3ft2XXL3p2/vOnJBz/Tj3JbsAuilnC4P90kv1Gm1wbRrTf/8R64Gf5hyUf//buNQd8lrc3rL8/3LXpdcXHbqiiEhpAp2XBiDqK5aeZKm2EetSBuGQDANtehyGltR9lAolWApLQYeiFTktE0AHFaBRwDjOIb3yAgpvNceLVIfCGZh5cfUdboIeQK9cxlVIQ8a8dCnYibnpmtvLj/wP6HhrY/e/eeqefv3/zwX05gciEGVwSvwtbKCkfi22BqxKEIF92y59sfG2G/fKrhliXnfWZx0HPK+nxb37rGpu61fnPPChIEvcQvtJMKbQAvKrCWycnPeRO5u+hQeNwgikjZQWz1RE2QgQXqmiQZklbgUvFOSQZIjddcPWFURLHwi53bBEqe781FNDc5O10aqcBUf2m6f3tlat+WqZn+F0Z2b3x+06avHQBYamtuVbUjdvbikfz4HDYrPHRP9zz6j3vZL//79bIL/qotv/rUE/Mrz1jX0HLSmmBgZFXkMfctgR72divj+HpuPgBZ4KId267RwKFJPJW95QAZBqDKSADDBgu4wQmWxINMaUXBHmjUEjzCbF9CpxhNY2zesD+fr98xUypuGxnavOXgyGMvTs4+tXnrY18dT1J8XSeVv0LDwBEO/xtI2LmADY0uOQAAAABJRU5ErkJggg=="
            )
            me.image(style=me.Style(width=75),
              src="data:image/png;base64,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"
            )
  
  # Members section
    with me.box(style=me.Style(align_self="stretch", background="white", justify_content="center", display="flex")):
      with me.box(style=me.Style(width="100%", padding=me.Padding(top=100, right=100, left=100, bottom=20), max_width=1440, flex_direction="column", justify_content="flex-start", align_items= "flex-start", gap=20, display="inline-flex")):
        me.text(text="About the Members", type="headline-3", 
            style = me.Style(font_weight = "bold", color ="Black", font_family = "Inter", margin=me.Margin.all(0)))
        me.text(text="We are Calvin Nguyen and Samantha Lin! We are the main builders of this tool, but everyone else within B01 of DSC 180AB helped as well. We are both data science majors and will be graduating Spring 2025. Please connect with us on Linkedin and you can see our prior work with our Github.", 
                style = me.Style(font_family="Inter"), type="body-1")
        with me.box(style=me.Style(align_self="stretch", justify_content="space-between", align_items="flex-start", gap=150, display="inline-flex")):
          with me.card(appearance="outlined", style=me.Style(font_family="Inter", border=me.Border.all(me.BorderSide(width="2px", color="#5271FF", style="solid")))):
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
          
          with me.card(appearance="outlined", style=me.Style(font_family="Inter",border=me.Border.all(me.BorderSide(width="2px", color="#5271FF", style="solid")))):
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
# Research Paper section
    with me.box(style=me.Style(align_self="stretch", background="white", justify_content="center", display="flex")):
      # flex_direction="column", 
      # justify_content="flex-start", align_items= "flex-start", gap=20, display="flex"
      with me.box(style=me.Style(width="100%", padding=me.Padding.symmetric(horizontal=100, vertical=20), max_width=1440, display='flex', flex_direction='column', gap=20, justify_content="flex-start")):
        me.text(text="Research Paper", type="headline-3", 
            style = me.Style(font_weight = "bold", color ="Black", font_family = "Inter", margin=me.Margin.all(0)))
        me.text(text="Below, you can read through the research paper. Feel free to download it as well!", 
                type="body-1", style = me.Style(font_family="Inter"))
        me.embed(
          src="https://olive-edie-30.tiiny.site/",
          style=me.Style(width="100%", height=800)
        )
        
@me.page(path = "/deep_analysis",stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap"
    ])
def deep_analysis_page():
  state = me.state(State)
  with me.box(style=me.Style(background="white", width="100%", display="flex", flex_direction="column", justify_content="flex-start", margin=me.Margin.all(0), overflow="auto")):
    # nav bar
    with me.box(style=me.Style(position='fixed', width="100%", display='flex', top=0, overflow='hidden', justify_content="space-between", align_content="center", background='white', border=me.Border(bottom=me.BorderSide(width="0.5px", color='#010021', style='solid')), padding=me.Padding.symmetric(vertical=15, horizontal=50), z_index=10)):
      me.html(
        """
        <a href="/">
          <img src="https://res.cloudinary.com/dd7kwlela/image/upload/v1738889378/capstone-dsc180b/jiz38dkxevducq0rpeye.png" alt="Home" height=48>
        </a>
        """
      )
      with me.box(style=me.Style(justify_content="flex-start", align_items="center", gap=40, display="flex")):
        me.link(text="Try Chenly Insights", url="/insights", style=me.Style(text_decoration='none', font_family='Inter', color="white", font_size=16, font_weight='bold', background="#010021", padding=me.Padding.symmetric(vertical=8, horizontal=10), border_radius=5))
        me.link(text="Prompt Testing", url="/prompt_testing", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
        me.link(text="Pipeline Explanation", url="/pipeline_explanation", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
        me.link(text="About Us", url="/about_us", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
    
    # header
    with me.box(style=me.Style(align_self="stretch", justify_content="center", display='flex', background= "linear-gradient(to right, #5271FF , #22BB7C)")):
      with me.box(style=me.Style(width="100%", max_width=1440, height="auto", padding = me.Padding.symmetric(vertical=100, horizontal=100),margin = me.Margin(top=80, bottom=10))):
        me.text(text="Chenly Insights - Deep Analysis", type="headline-2", style = me.Style(font_weight = "bold", color ="white", font_family = "Inter", margin=me.Margin.all(0)))
    with me.box(style=me.Style(align_self="stretch", background="white", justify_content="center", display="flex")):
      with me.box(style=me.Style(width="100%", padding=me.Padding(top=50, right=100, left=100, bottom=50), max_width=1440, flex_direction="column", justify_content="flex-start", align_items= "flex-start", gap=40, display="inline-flex")):
        # if you want to do score with special text colors, make this into a me.box with two different texts 
        me.text("Score: " + f"{state.veracity}, {state.veracity_label}", type='headline-3', style = me.Style(font_weight = "bold", color ="#010021", font_family = "Inter", margin=me.Margin.all(0)))
        with me.box(style=me.Style(width="100%", justify_content='flex-start', align_items='flex-start', gap=30, display='flex', flex_wrap='wrap')):
          # article title
          with me.box(style=me.Style(padding=me.Padding.symmetric(horizontal=5, vertical=10), border_radius="5px", border=me.Border.all(me.BorderSide(width="1.5px", color="#010021", style='solid')), display='flex', flex_direction="column", justify_content="flex-start", align_items="flex-start", gap=5, max_width=350)):
            me.text("Article Title", style=me.Style(font_size=20, font_weight="bold", font_family="Inter"))
            me.box(style=me.Style(align_self='stretch', height=0, border=me.Border.all(me.BorderSide(width="0.5px", color="#010021", style='solid'))))
            me.text(f"{state.article_title}", style=me.Style(font_size=16, font_family="Inter"))
          # author
          with me.box(style=me.Style(padding=me.Padding.symmetric(horizontal=5, vertical=10), border_radius="5px", border=me.Border.all(me.BorderSide(width="1.5px", color="#010021", style='solid')), display='flex', flex_direction="column", justify_content="flex-start", align_items="flex-start", gap=5, max_width=350)):
            me.text("Author", style=me.Style(font_size=20, font_weight="bold", font_family="Inter"))
            me.box(style=me.Style(align_self='stretch', height=0, border=me.Border.all(me.BorderSide(width="0.5px", color="#010021", style='solid'))))
            me.text(f"{state.article_author}", style=me.Style(font_size=16, font_family="Inter"))
          # Date
          with me.box(style=me.Style(padding=me.Padding.symmetric(horizontal=5, vertical=10), border_radius="5px", border=me.Border.all(me.BorderSide(width="1.5px", color="#010021", style='solid')), display='flex', flex_direction="column", justify_content="flex-start", align_items="flex-start", gap=5, max_width=350)):
            me.text("Date", style=me.Style(font_size=20, font_weight="bold", font_family="Inter"))
            me.box(style=me.Style(align_self='stretch', height=0, border=me.Border.all(me.BorderSide(width="0.5px", color="#010021", style='solid'))))
            me.text(f"{state.article_date}", style=me.Style(font_size=16, font_family="Inter"))
          # Source
          with me.box(style=me.Style(padding=me.Padding.symmetric(horizontal=5, vertical=10), border_radius="5px", border=me.Border.all(me.BorderSide(width="1.5px", color="#010021", style='solid')), display='flex', flex_direction="column", justify_content="flex-start", align_items="flex-start", gap=5, max_width=350)):
            me.text("Source", style=me.Style(font_size=20, font_weight="bold", font_family="Inter"))
            me.box(style=me.Style(align_self='stretch', height=0, border=me.Border.all(me.BorderSide(width="0.5px", color="#010021", style='solid'))))
            me.text(f"{state.article_source}", style=me.Style(font_size=16, font_family="Inter"))
          # Topic
          with me.box(style=me.Style(padding=me.Padding.symmetric(horizontal=5, vertical=10), border_radius="5px", border=me.Border.all(me.BorderSide(width="1.5px", color="#010021", style='solid')), display='flex', flex_direction="column", justify_content="flex-start", align_items="flex-start", gap=5, max_width=350)):
            me.text("Topic", style=me.Style(font_size=20, font_weight="bold", font_family="Inter"))
            me.box(style=me.Style(align_self='stretch', height=0, border=me.Border.all(me.BorderSide(width="0.5px", color="#010021", style='solid'))))
            me.text(f"{state.article_topic}", style=me.Style(font_size=16, font_family="Inter"))
        with me.box(style=me.Style(width="100%", justify_content='space-around', align_items='flex-start', gap=30, display='flex')):
          # plot
          categories = ["Stance Detection", "Naive Realism", "Sensationalism", "Social Credibility"]
          values = [state.overall_stance_score, state.overall_naive_realism_score, state.overall_sens_score, state.overall_social_credibility] 
          num_vars = len(categories)
          angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
          values += values[:1]  
          angles += angles[:1]  
          fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
          ax.set_xticks(angles[:-1])
          ax.set_xticklabels(categories)
          ax.set_ylim(0, 6)
          ax.plot(angles, values, color='b', linewidth=2, linestyle='solid')
          ax.fill(angles, values, color='b', alpha=0.3)
          ax.set_yticks(range(0, 7))
          ax.set_yticklabels(map(str, range(0, 7)))
          plt.title("Radar Chart for Article Analysis")
          
          me.plot(fig, style=me.Style(width="100%"))
          # summary
          with me.box(style=me.Style(padding=me.Padding.symmetric(horizontal=5, vertical=10), border_radius="5px", border=me.Border.all(me.BorderSide(width="1.5px", color="#010021", style='solid')), display='flex', flex_direction="column", justify_content="flex-start", align_items="flex-start", gap=5, max_width=700, width="50%")):
            me.text("Summary", style=me.Style(font_size=24, font_weight="bold", font_family="Inter"))
            me.box(style=me.Style(align_self='stretch', height=0, border=me.Border.all(me.BorderSide(width="0.5px", color="#010021", style='solid'))))
            me.markdown(f"{state.article_summary}", style=me.Style(font_family="Inter", font_size=16, color="#010021", margin=me.Margin.symmetric(vertical=0)))
        me.text("Score Table", type='headline-3', style = me.Style(font_weight = "bold", color ="#010021", font_family = "Inter", margin=me.Margin.all(0)))
        def get_data_for_table(prompting_selection, adjustments, factuality_factors):
          if prompting_selection == "FCOT":
            score_dict = {"Sensationalism": str(round(float(state.overall_sens_score),2)), "Political_stance": str(round(float(state.overall_stance_score),2)),
                          "Naive_realism": str(round(float(state.overall_naive_realism_score),2)), "Social_credibility": str(round(float(state.overall_social_credibility),2))}
            consideration_dict = {"Sensationalism": state.fcot_response_dict.get('Sensationalism'), "Political_stance": state.fcot_response_dict.get('Political_stance'),
                          "Naive_realism": "N/A calculated by Predictive AI", "Social_credibility": "N/A calculated by Predictive AI"}
            if "SERP_API" in adjustments:
              citation_dict = {"Sensationalism": "gemini-1.5-pro-002, Serp API search result", "Political_stance": "gemini-1.5-pro-002, Serp API search result","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
            elif "Vector_Database" in adjustments:
              citation_dict = {"Sensationalism": "gemini-1.5-pro-002, Politifact, Snopes", "Political_stance": "gemini-1.5-pro-002, Politifact, Snopes","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
            elif "SERP_API" in adjustments and "Vector_Database" in adjustments:
              citation_dict = {"Sensationalism": "gemini-1.5-pro-002, Serp API search result, Politifact, Snopes", "Political_stance": "gemini-1.5-pro-002, Serp API search result, Politifact, Snopes","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
            else:
              citation_dict = {"Sensationalism": "gemini-1.5-pro-002", "Political_stance": "gemini-1.5-pro-002","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
          elif prompting_selection == "Normal":
            score_dict = {"Sensationalism": str(round(float(state.overall_sens_score),2)), "Political_stance": str(round(float(state.overall_stance_score),2)),
                          "Naive_realism": str(round(float(state.overall_naive_realism_score),2)), "Social_credibility": str(round(float(state.overall_social_credibility),2))}
            consideration_dict = {"Sensationalism": state.normal_response_dict.get('Sensationalism'), "Political_stance": state.normal_response_dict.get('Political_stance'),
                          "Naive_realism": "N/A calculated by Predictive AI", "Social_credibility": "N/A calculated by Predictive AI"}
            if "SERP_API" in adjustments:
              citation_dict = {"Sensationalism": "gemini-1.5-pro-002, Serp API search result", "Political_stance": "gemini-1.5-pro-002, Serp API search result","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
            elif "Vector_Database" in adjustments:
              citation_dict = {"Sensationalism": "gemini-1.5-pro-002, Politifact, Snopes", "Political_stance": "gemini-1.5-pro-002, Politifact, Snopes","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
            elif "SERP_API" in adjustments and "Vector_Database" in adjustments:
              citation_dict = {"Sensationalism": "gemini-1.5-pro-002, Serp API search result, Politifact, Snopes", "Political_stance": "gemini-1.5-pro-002, Serp API search result, Politifact, Snopes","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
            else:
              citation_dict = {"Sensationalism": "gemini-1.5-pro-002", "Political_stance": "gemini-1.5-pro-002","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
          else:
            score_dict = {"Sensationalism": str(round(float(state.overall_sens_score),2)), "Political_stance": str(round(float(state.overall_stance_score),2)),
                          "Naive_realism": str(round(float(state.overall_naive_realism_score),2)), "Social_credibility": str(round(float(state.overall_social_credibility),2))}
            consideration_dict = {"Sensationalism": state.cot_response_dict.get('Sensationalism'), "Political_stance": state.cot_response_dict.get('Political_stance'),
                          "Naive_realism": "N/A calculated by Predictive AI", "Social_credibility": "N/A calculated by Predictive AI"}
            if "SERP_API" in adjustments:
              citation_dict = {"Sensationalism": "gemini-1.5-pro-002, Serp API search result", "Political_stance": "gemini-1.5-pro-002, Serp API search result","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
            elif "Vector_Database" in adjustments:
              citation_dict = {"Sensationalism": "gemini-1.5-pro-002, Politifact, Snopes", "Political_stance": "gemini-1.5-pro-002, Politifact, Snopes","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
            elif "SERP_API" in adjustments and "Vector_Database" in adjustments:
              citation_dict = {"Sensationalism": "gemini-1.5-pro-002, Serp API search result, Politifact, Snopes", "Political_stance": "gemini-1.5-pro-002, Serp API search result, Politifact, Snopes","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
            else:
              citation_dict = {"Sensationalism": "gemini-1.5-pro-002", "Political_stance": "gemini-1.5-pro-002","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
          data = {
            "FACTUALITY FACTOR": factuality_factors,
            "SCORE": [score_dict[f] for f in factuality_factors],
            "CONSIDERATION": [consideration_dict[f] for f in factuality_factors],
            "CITATION": [citation_dict[f] for f in factuality_factors]
            }
          return pd.DataFrame(data)

        # score_table = pd.DataFrame(
        #   data = {
        #     "FACTUALITY FACTOR": ["Sensationalism", "Political Stance", "Naive Realism", "Social Credibility"],
        #     "SCORE": [str(round(float(state.overall_sens_fcot_score),2)), str(round(float(state.overall_stance_fcot_score),2)), str(round(float(state.overall_naive_realism_score),2)), str(round(float(state.overall_social_credibility),2))],
        #     "CONSIDERATION": [state.fcot_response_dict.get('Sensationalism'), state.fcot_response_dict.get('Political_stance'), "N/A calculated by Predictive AI", "N/A calculated by Predictive AI"],
        #     "CITATION": ["gemini-1.5-pro-002",'gemini-1.5-pro-002','Liar Plus dataset, XGBoost Tree', 'Liar Plus dataset, Pytorch Neural Network']
        #   }
        # )

        with me.box(style=me.Style(padding=me.Padding.all(15), margin=me.Margin.all(15))):
          prompting_selection = state.radio_value
          adjustments = state.toggle_values
          factuality_factors = state.selected_values_1
          print("generate table")
          me.table(get_data_for_table(prompting_selection, adjustments, factuality_factors))

        with me.box(style=me.Style(height=600, width="100%")):
          mel.chat(
            transform, 
            title="Gemini Misinformation Helper", 
            bot_user="Chenly", # Short for the Vietnamese word for Truth
          )

# starting page
@me.page(path='/insights',stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap"
    ])
def insights():
  with me.box(style=me.Style(background="white", width="100%", display="flex", flex_direction="column", justify_content="flex-start", margin=me.Margin.all(0), overflow="auto")):
    # navbar
    with me.box(style=me.Style(position='fixed', width="100%", display='flex', top=0, overflow='hidden', justify_content="space-between", align_content="center", background='white', border=me.Border(bottom=me.BorderSide(width="0.5px", color='#010021', style='solid')), padding=me.Padding.symmetric(vertical=15, horizontal=50), z_index=10)):
      me.html(
        """
        <a href="/">
          <img src="https://res.cloudinary.com/dd7kwlela/image/upload/v1738889378/capstone-dsc180b/jiz38dkxevducq0rpeye.png" alt="Home" height=48>
        </a>
        """
      )
      with me.box(style=me.Style(justify_content="flex-start", align_items="center", gap=40, display="flex")):
        me.link(text="Try Chenly Insights", url="/insights", style=me.Style(text_decoration='none', font_family='Inter', color="white", font_size=16, font_weight='bold', background="#010021", padding=me.Padding.symmetric(vertical=8, horizontal=10), border_radius=5))
        me.link(text="Prompt Testing", url="/prompt_testing", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
        me.link(text="Pipeline Explanation", url="/pipeline_explanation", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
        me.link(text="About Us", url="/about_us", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
    
    # second header
    with me.box(style=me.Style(align_self="stretch", justify_content="center", display='flex', background= "linear-gradient(to right, #5271FF , #22BB7C)")):
      with me.box(style=me.Style(width="100%", max_width=1440, height="auto", padding = me.Padding.symmetric(vertical=100, horizontal=100),margin = me.Margin(top=80, bottom=10))):
        me.text(text="Chenly Insights - Selection", type="headline-2", style = me.Style(font_weight = "bold", color ="white", font_family = "Inter", margin=me.Margin.all(0)))
    
    # Select your setting section
    with me.box(style=me.Style(align_self="stretch", background="white", justify_content="center", display="flex")):
      with me.box(style=me.Style(width="100%", max_width=1440, background='white', padding=me.Padding.symmetric(horizontal=100, vertical=70), flex_direction='column', justify_content='center', align_content='center', display='flex', gap=10, min_height=750)):
        with me.box(style=me.Style(align_self='stretch', flex="1 1 0", padding=me.Padding.all(50), border_radius=10, border=me.Border.all(me.BorderSide(width=1, color="#010021", style='solid')), flex_direction='column', justify_content='flex-start', align_items="flex-start", gap=50, display='flex')):
          me.text(text = "Select your Settings: ", type = "headline-5", 
                style = me.Style(font_weight = "bold", color ="Black", font_family = "Inter", margin=me.Margin.all(0)))
          me.text(text = "Recommended settings contain the best prompting types, adjustments, and all of the factuality factors available. If you would like to adjust these settings, plice click \"create adjustment\".", type = "body-1",
                style=me.Style(font_family="Inter", color="black"))
        # justify_content="flex-start", align_items="center", gap=5, display="inline-flex")
          with me.box(style=me.Style(align_self="stretch",justify_content="center", align_items="center", gap=28, display="inline-flex")):
            def recommended_selection(event: me.ClickEvent):
              state = me.state(State)
              state.selected_values_1 = ["Social_credibility", "Naive_realism", "Sensationalism", "Political_stance"]
              state.radio_value = 'FCOT'
              state.toggle_values = ["Vector_Database", "SERP_API", "Function_Call"] 
              me.navigate("/uploadpdf")
            me.button(label="Recommended", type='flat', on_click=recommended_selection, style=me.Style(font_family="Inter", font_size=16, font_style='bold', background="5271FF", border_radius=5))
            def navigate_to_ca(event: me.ClickEvent):
              me.navigate('/adjusting')
            me.button(label="Create Adjustments", type='flat', on_click=navigate_to_ca, style=me.Style(font_family="Inter", background="#A5A5A3", font_size=16, font_style='bold', border_radius=5))
            # me.link(text="Create Adjustments", url="/adjusting", style=me.Style(text_decoration='none', font_family='Inter', color="white", font_size=16, font_weight='bold', background="#A5A5A3", padding=me.Padding.symmetric(vertical=8, horizontal=10), border_radius=5))

def on_selection_change_1(e: me.SelectSelectionChangeEvent):
  state = me.state(State)
  state.selected_values_1 = e.values
  print(state.selected_values_1)

def on_toggle_change(e: me.SelectSelectionChangeEvent):
  state = me.state(State)
  state.toggle_values = e.values

def on_change(event: me.RadioChangeEvent):
  state = me.state(State)
  state.radio_value = event.value

@me.page(path='/adjusting', stylesheets=[
  "https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap"
])
def adjusting():
  
  state = me.state(State)
  
  with me.box(style=me.Style(background="white", width="100%", display="flex", flex_direction="column", justify_content="flex-start", margin=me.Margin.all(0), overflow="auto")):
    # navbar
    with me.box(style=me.Style(position='fixed', width="100%", display='flex', top=0, overflow='hidden', justify_content="space-between", align_content="center", background='white', border=me.Border(bottom=me.BorderSide(width="0.5px", color='#010021', style='solid')), padding=me.Padding.symmetric(vertical=15, horizontal=50), z_index=10)):
      me.html(
        """
        <a href="/">
          <img src="https://res.cloudinary.com/dd7kwlela/image/upload/v1738889378/capstone-dsc180b/jiz38dkxevducq0rpeye.png" alt="Home" height=48>
        </a>
        """
      )
      with me.box(style=me.Style(justify_content="flex-start", align_items="center", gap=40, display="flex")):
        me.link(text="Try Chenly Insights", url="/insights", style=me.Style(text_decoration='none', font_family='Inter', color="white", font_size=16, font_weight='bold', background="#010021", padding=me.Padding.symmetric(vertical=8, horizontal=10), border_radius=5))
        me.link(text="Prompt Testing", url="/prompt_testing", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
        me.link(text="Pipeline Explanation", url="/pipeline_explanation", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
        me.link(text="About Us", url="/about_us", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))

    # second header
    with me.box(style=me.Style(align_self="stretch", justify_content="center", display='flex', background= "linear-gradient(to right, #5271FF , #22BB7C)")):
      with me.box(style=me.Style(width="100%", max_width=1440, height="auto", padding = me.Padding.symmetric(vertical=100, horizontal=100),margin = me.Margin(top=80, bottom=10))):
        me.text(text="Chenly Insights - Adjustment", type="headline-2", style = me.Style(font_weight = "bold", color ="white", font_family = "Inter", margin=me.Margin.all(0)))
    
    # Select your adjustment section
    with me.box(style=me.Style(align_self="stretch", background="white", justify_content="center", display="flex")):
      with me.box(style=me.Style(width="100%", max_width=1440, background='white', padding=me.Padding.symmetric(horizontal=100, vertical=70), flex_direction='column', justify_content='center', align_content='center', display='flex', gap=10, min_height=750)):
        with me.box(style=me.Style(align_self='stretch', flex="1 1 0", padding=me.Padding.all(50), border_radius=10, border=me.Border.all(me.BorderSide(width=1, color="#010021", style='solid')), flex_direction='column', justify_content='flex-start', align_items="left", gap=50, display='flex')):
          with me.box(style=me.Style(flex_direction='column', justify_content='flex-start', align_items='flex-start', display='flex', gap="18.75px")):
            me.text(text = "Select your Prompting Techinique:", type = "headline-5", style = me.Style(font_weight = "bold", color ="Black", font_family = "Inter", margin=me.Margin.all(0)))
            with me.box(style=me.Style(justify_content="flex-start", align_items="left", gap=35, display="inline-flex", background = "white")):
              # should replace it with a radio button
              me.radio(
                on_change=on_change,
                options=[
                  me.RadioOption(label="Normal", value="Normal"),
                  me.RadioOption(label="COT", value="COT"),
                  me.RadioOption(label="FCOT", value="FCOT"),
                ],
                value=state.radio_value,
              )
              # me.button(label="Normal", type="stroked", style=me.Style(font_family="Inter", background="white", color="black"))
              # me.button(label="COT", type="stroked", style=me.Style(font_family="Inter", background="white", color="black"))
              # me.button(label="FCOT", type="stroked", style=me.Style(font_family="Inter", background="white", color="black"))
          # select Adjustments
          with me.box(style=me.Style(flex_direction='column', justify_content='flex-start', align_items='flex-start', display='flex', gap="18.75px")):
            me.text(text = "Select your Adjustments:", type = "headline-5", style = me.Style(font_weight = "bold", color ="Black", font_family = "Inter", margin=me.Margin.all(0)))
            me.button_toggle(
              buttons=[
                me.ButtonToggleButton(label="Vector Database", value="Vector_Database"),
                me.ButtonToggleButton(label="SERP API", value="SERP_API"),
                me.ButtonToggleButton(label="Function Call", value="Function_Call"),
              ],
              on_change=on_toggle_change,
              multiple=True,
              hide_selection_indicator=False,
              disabled=False,
              value=state.toggle_values,
              style=me.Style(font_family="Inter", margin=me.Margin.symmetric(horizontal=10), background="white")# blue = #5271FF
            )
          # select factuality factors
          with me.box(style=me.Style(flex_direction='column', justify_content='flex-start', align_items='flex-start', display='flex', gap="18.75px")):
            me.text(text = "Select your Factuality Factors", type = "headline-5", style = me.Style(font_weight = "bold", color ="Black", font_family = "Inter", margin=me.Margin.all(0)))
            me.select(
                label="Select multiple",
                options=[
                  me.SelectOption(label="Social Credibility", value="Social_credibility"),
                  me.SelectOption(label="Naive Realism", value="Naive_realism"),
                  me.SelectOption(label="Sensationalism", value="Sensationalism"),
                  me.SelectOption(label="Stance Detection", value="Political_stance")
                ],
                on_selection_change=on_selection_change_1,
                style=me.Style(width=500),
                multiple=True,
                appearance="outline",
                value=state.selected_values_1,
            )
            me.text(
              # "something should be showing here but its not working rn"
              text="Selected values (multiple): " + ", ".join(state.selected_values_1), type = "subtitle-1"
            )
          # confirm button
          def save_selections(event: me.ClickEvent):
            if (state.radio_value == '') or (len(state.selected_values_1) == 0):
              print('please complete selection')
            else:
              me.navigate("/uploadpdf")
          me.button(label="Confirm", on_click=save_selections, style=me.Style(width="100%",  font_family='Inter', color="white", font_size=35, font_weight='bold', background="#22BB7C", padding=me.Padding.symmetric(vertical=10), border_radius=10, display='flex', justify_content='center'))

def process_submission(event: me.ClickEvent):
  # will need to add stuff to actually analyze the article
  state = me.state(State)
  if state.uploaded == False:
    convert_url_to_pdf(state.link, r"C:\temp\analyze.pdf")
    with open(r"C:\temp\analyze.pdf", 'rb') as f:
      state.file = me.UploadedFile(contents=f.read())
    state.uploaded = True
    get_metadata(state.chat_history)
  if state.radio_value == "Normal":
    ask_normal_prompting_questions(me.ClickEvent(key="normal_prompt", is_target=True))
  elif state.radio_value == "COT":
    ask_cot_prompting_questions(me.ClickEvent(key="cot_prompt", is_target=True))
  else: 
    ask_fcot_prompting_questions(me.ClickEvent(key="fcot_prompt", is_target=True))
  ask_pred_ai()
  state.finish_analysis = True

def link_inputted(e: me.InputBlurEvent):
  state = me.state(State)
  state.link = e.value

@me.page(path='/uploadpdf', stylesheets=[
  "https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap"
  # if coming here from "clear article button", we need a on_load func to reset the state variables
])
def uploadpdf():
  s = me.state(State)
  
  with me.box(style=me.Style(background="white", width="100%", display="flex", flex_direction="column", justify_content="flex-start", margin=me.Margin.all(0), overflow="auto")):
    # navbar
    with me.box(style=me.Style(position='fixed', width="100%", display='flex', top=0, overflow='hidden', justify_content="space-between", align_content="center", background='white', border=me.Border(bottom=me.BorderSide(width="0.5px", color='#010021', style='solid')), padding=me.Padding.symmetric(vertical=15, horizontal=50), z_index=10)):
      me.html(
        """
        <a href="/">
          <img src="https://res.cloudinary.com/dd7kwlela/image/upload/v1738889378/capstone-dsc180b/jiz38dkxevducq0rpeye.png" alt="Home" height=48>
        </a>
        """
      )
      with me.box(style=me.Style(justify_content="flex-start", align_items="center", gap=40, display="flex")):
        me.link(text="Try Chenly Insights", url="/insights", style=me.Style(text_decoration='none', font_family='Inter', color="white", font_size=16, font_weight='bold', background="#010021", padding=me.Padding.symmetric(vertical=8, horizontal=10), border_radius=5))
        me.link(text="Prompt Testing", url="/prompt_testing", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
        me.link(text="Pipeline Explanation", url="/pipeline_explanation", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))
        me.link(text="About Us", url="/about_us", style=me.Style(text_decoration='none', font_family='Inter', color="#010021", font_size=16, font_weight='bold'))

    # second header
    with me.box(style=me.Style(align_self="stretch", justify_content="center", display='flex', background= "linear-gradient(to right, #5271FF , #22BB7C)")):
      with me.box(style=me.Style(width="100%", max_width=1440, height="auto", padding = me.Padding.symmetric(vertical=100, horizontal=100),margin = me.Margin(top=80, bottom=10))):
        me.text(text="Chenly Insights - PDF", type="headline-2", style = me.Style(font_weight = "bold", color ="white", font_family = "Inter", margin=me.Margin.all(0)))

    with me.box(style=me.Style(align_self="stretch", background="white", justify_content="center", display="flex")):
      with me.box(style=me.Style(width="100%", max_width=1440, background='white', padding=me.Padding.symmetric(horizontal=100, vertical=70), flex_direction='column', justify_content='center', align_content='center', display='flex', gap=10, min_height=750)):
        with me.box(style=me.Style(align_self='stretch', flex="1 1 0", padding=me.Padding.all(50), border_radius=10, border=me.Border.all(me.BorderSide(width=1, color="#010021", style='solid')), flex_direction='column', justify_content='flex-start', align_items="left", gap=50, display='flex')):
          with me.box(style=me.Style(flex_direction='column', justify_content='flex-start', align_items='flex-start', display='flex', gap="18.75px")):
            with me.box(style=me.Style(align_self='stretch', justify_content='space-between', align_items='flex-start', display='flex')):
              with me.box(style=me.Style(justify_content='flex-start', align_items='flex-start', gap=28, display='flex')):
                with me.content_uploader(
                  accepted_file_types=["pdf"],
                  on_upload=handle_upload,
                  type="flat",
                  color="primary",
                  style=me.Style(font_weight="bold", background="#5271FF", height=50),
                ):
                  with me.box(style=me.Style(display="flex", gap=5)):
                    me.icon("upload")
                    me.text("Upload PDF", style=me.Style(font_size=20, font_family="Inter"))
                me.input(label="Link input", appearance="outline", on_blur=link_inputted, style=me.Style(width = "300px", margin=me.Margin.all(0), display='flex', justify_content='center'))
                me.button(label="Submit", on_click=process_submission, type='flat', style=me.Style(font_family="Inter", color="white", font_size=20, font_weight='bold', background="#5271FF", border_radius=5, padding=me.Padding.symmetric(vertical=5, horizontal=10), height=50))
              with me.box(style=me.Style(height=50, justify_content="center", align_items="center", display="flex", border=me.Border.all(me.BorderSide(width=1, color="#5271FF", style='solid')), padding=me.Padding.symmetric(vertical=5, horizontal=10), border_radius=5)):
                me.button(label="Make Adjustments", on_click = navigate_to_adjustment, type='flat', style=me.Style(font_family="Inter", color="#5271FF", font_size=20, font_weight='bold', background="white", border_radius=5, padding=me.Padding.symmetric(vertical=5, horizontal=10), height=40))
          if s.finish_analysis == True: 
            with me.box(style = me.Style(flex_direction='column', justify_content='flex-start', align_items='center', gap=50, display='flex')):
              with me.box(style=me.Style(display='flex', justify_content='center', align_items='flex-start', gap=65)):
                with me.box(style=me.Style(flex_direction='column', justify_content='flex-start', align_items='flex-start', gap=5, display='flex')):
                  me.text("This article is found to be", type="headline-4", style=me.Style(color="#DA5D39", font_family="Inter", margin=me.Margin.all(0)))
                  me.text(f"{s.veracity_label}", type="headline-4", style=me.Style(color="#DA5D39", font_family="Inter", font_weight='bold', margin=me.Margin.all(0)))
              if s.veracity_label == "Pants on Fire":
                me.icon(icon="emergency_heat", style=me.Style(color="#DA5D39", font_size="64px",width=64, height=64))
              elif s.veracity_label == "False":
                me.icon(icon="close", style=me.Style(color="#DA5D39", font_size="64px",width=64, height=64)) # false
              elif s.veracity_label == "Barely True":
                me.icon(icon="transition_fade", style=me.Style(color="#FDB815", font_size="64px",width=64, height=64)) # barely true
              elif s.veracity_label == "Half True":
                me.icon(icon="star_rate_half", style=me.Style(color="#FDB815", font_size="64px",width=64, height=64)) # half true
              elif s.veracity_label == "Mostly True":
                me.icon(icon="check", style=me.Style(color="#22BB7C", font_size="64px",width=64, height=64)) # mostly true
              elif s.veracity_label == "True":
                me.icon(icon="done_all", style=me.Style(color="#22BB7C", font_size="64px",width=64, height=64)) # true
              with me.box(style = me.Style(align_self='stretch')):
                me.progress_bar(color='warn', value=s.veracity * (100/6), mode='determinate') # use accent for mostlytrue and true, warn for false and pants on fire, primary for barely true and half true
            with me.box(style=me.Style(display='flex', justify_content='center', align_content='flex-start', gap=28)):
              me.button(label="Deep Analysis (Highly Recommend)", on_click=navigate_to_deep, type="flat", style=me.Style(height=50, background="linear-gradient(to right, #5271FF , #22BB7C)", padding=me.Padding.symmetric(vertical=5, horizontal=10), border_radius=5, font_family="Inter", color="white", font_size=20, font_weight="bold"))
              me.button(label="Clear Article", type="flat", on_click=reset_article, style=me.Style(height=50, background="#5271FF", border_radius=5, padding=me.Padding.symmetric(vertical=5, horizontal=10), font_family="Inter", color="white", font_size=20, font_weight="bold"))

def navigate_to_adjustment(e: me.ClickEvent):
  me.navigate("/adjustment")

def navigate_to_deep(e: me.ClickEvent):
  me.navigate("/deep_analysis")

def reset_article(e: me.ClickEvent):
  state = me.state(State)
  state.article_title = ""
  state.chat_history = []
  state.veracity = 0.0
  state.veracity_label = ""
  state.link = ""
  state.finish_analysis = False
  state.vdb_response = ""
  state.serp_response = ""
  state.overall_sens_score = 0
  state.overall_stance_score = 0
  state.overall_social_credibility = 0
  state.overall_naive_realism_score = 0
  me.navigate('/uploadpdf')
            

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
          label="Ask cot Prompt Questions",
          on_click=ask_cot_prompting_questions,
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
    def get_data_for_table(prompting_selection, adjustments, factuality_factors):
      if prompting_selection == "FCOT":
        score_dict = {"Sensationalism": str(round(float(state.overall_sens_fcot_score),2)), "Political_stance": str(round(float(state.overall_stance_fcot_score),2)),
                      "Naive_realism": str(round(float(state.overall_naive_realism_score),2)), "Social_credibility": str(round(float(state.overall_social_credibility),2))}
        consideration_dict = {"Sensationalism": state.fcot_response_dict.get('Sensationalism'), "Political_stance": state.fcot_response_dict.get('Political_stance'),
                      "Naive_realism": "N/A calculated by Predictive AI", "Social_credibility": "N/A calculated by Predictive AI"}
        if "SERP_API" in adjustments:
          citation_dict = {"Sensationalism": "gemini-1.5-pro-002, Serp API search result", "Political_stance": "gemini-1.5-pro-002, Serp API search result","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
        elif "Vector_Database" in adjustments:
          citation_dict = {"Sensationalism": "gemini-1.5-pro-002, Politifact, Snopes", "Political_stance": "gemini-1.5-pro-002, Politifact, Snopes","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
        elif "SERP_API" in adjustments and "Vector_Database" in adjustments:
          citation_dict = {"Sensationalism": "gemini-1.5-pro-002, Serp API search result, Politifact, Snopes", "Political_stance": "gemini-1.5-pro-002, Serp API search result, Politifact, Snopes","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
        else:
          citation_dict = {"Sensationalism": "gemini-1.5-pro-002", "Political_stance": "gemini-1.5-pro-002","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
      elif prompting_selection == "Normal":
        score_dict = {"Sensationalism": str(round(float(state.overall_sens_normal_score),2)), "Political_stance": str(round(float(state.overall_stance_normal_score),2)),
                      "Naive_realism": str(round(float(state.overall_naive_realism_score),2)), "Social_credibility": str(round(float(state.overall_social_credibility),2))}
        consideration_dict = {"Sensationalism": state.normal_response_dict.get('Sensationalism'), "Political_stance": state.normal_response_dict.get('Political_stance'),
                      "Naive_realism": "N/A calculated by Predictive AI", "Social_credibility": "N/A calculated by Predictive AI"}
        if "SERP_API" in adjustments:
          citation_dict = {"Sensationalism": "gemini-1.5-pro-002, Serp API search result", "Political_stance": "gemini-1.5-pro-002, Serp API search result","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
        elif "Vector_Database" in adjustments:
          citation_dict = {"Sensationalism": "gemini-1.5-pro-002, Politifact, Snopes", "Political_stance": "gemini-1.5-pro-002, Politifact, Snopes","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
        elif "SERP_API" in adjustments and "Vector_Database" in adjustments:
          citation_dict = {"Sensationalism": "gemini-1.5-pro-002, Serp API search result, Politifact, Snopes", "Political_stance": "gemini-1.5-pro-002, Serp API search result, Politifact, Snopes","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
        else:
          citation_dict = {"Sensationalism": "gemini-1.5-pro-002", "Political_stance": "gemini-1.5-pro-002","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
      else:
        score_dict = {"Sensationalism": str(round(float(state.overall_sens_cot_score),2)), "Political_stance": str(round(float(state.overall_stance_cot_score),2)),
                      "Naive_realism": str(round(float(state.overall_naive_realism_score),2)), "Social_credibility": str(round(float(state.overall_social_credibility),2))}
        consideration_dict = {"Sensationalism": state.cot_response_dict.get('Sensationalism'), "Political_stance": state.cot_response_dict.get('Political_stance'),
                      "Naive_realism": "N/A calculated by Predictive AI", "Social_credibility": "N/A calculated by Predictive AI"}
        if "SERP_API" in adjustments:
          citation_dict = {"Sensationalism": "gemini-1.5-pro-002, Serp API search result", "Political_stance": "gemini-1.5-pro-002, Serp API search result","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
        elif "Vector_Database" in adjustments:
          citation_dict = {"Sensationalism": "gemini-1.5-pro-002, Politifact, Snopes", "Political_stance": "gemini-1.5-pro-002, Politifact, Snopes","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
        elif "SERP_API" in adjustments and "Vector_Database" in adjustments:
          citation_dict = {"Sensationalism": "gemini-1.5-pro-002, Serp API search result, Politifact, Snopes", "Political_stance": "gemini-1.5-pro-002, Serp API search result, Politifact, Snopes","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
        else:
          citation_dict = {"Sensationalism": "gemini-1.5-pro-002", "Political_stance": "gemini-1.5-pro-002","Naive_realism": 'Liar Plus dataset, XGBoost Tree', "Social_credibility": 'Liar Plus dataset, Pytorch Neural Network'}
      data = {
        "FACTUALITY FACTOR": factuality_factors,
        "SCORE": [score_dict[f] for f in factuality_factors],
        "CONSIDERATION": [consideration_dict[f] for f in factuality_factors],
        "CITATION": [citation_dict[f] for f in factuality_factors]
        }
      return pd.DataFrame(data)

    # score_table = pd.DataFrame(
    #   data = {
    #     "FACTUALITY FACTOR": ["Sensationalism", "Political Stance", "Naive Realism", "Social Credibility"],
    #     "SCORE": [str(round(float(state.overall_sens_fcot_score),2)), str(round(float(state.overall_stance_fcot_score),2)), str(round(float(state.overall_naive_realism_score),2)), str(round(float(state.overall_social_credibility),2))],
    #     "CONSIDERATION": [state.fcot_response_dict.get('Sensationalism'), state.fcot_response_dict.get('Political_stance'), "N/A calculated by Predictive AI", "N/A calculated by Predictive AI"],
    #     "CITATION": ["gemini-1.5-pro-002",'gemini-1.5-pro-002','Liar Plus dataset, XGBoost Tree', 'Liar Plus dataset, Pytorch Neural Network']
    #   }
    # )

    with me.box(style=me.Style(padding=me.Padding.all(15), margin=me.Margin.all(15))):
      prompting_selection = state.radio_value
      adjustments = state.toggle_values
      factuality_factors = state.selected_values_1
      print("generate table")
      me.table(get_data_for_table(prompting_selection, adjustments, factuality_factors))

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
def get_metadata(history: list[mel.ChatMessage]):
  """asks gemini to go through the text of the pdf uploaded by the user and get the headline of the text 
    
    Args:
        history: chat history 
  """
  # need to adjust this to get headline, author, date, keywords, and source, and summary
  state= me.state(State)
  chat_history = ""
  if state.file and state.uploaded:
    chat_history += f"\nuser: {pdf_to_text(state.file)}"
  chat_history += "\n".join(f"{message.role}: {message.content}" for message in history)
  full_input = f"{chat_history}\n"
  user_input = """
Please extract the headline, author, date, keywords, source, and summary of this article.
It is very import to format the information like so:
headline: {insert headline}
date: {insert date}
author: {insert author}
keywords: {insert keywords}
source: {insert source}
summary: {insert summary}
"""
  full_input = full_input + user_input
  time.sleep(2)
  response = model.generate_content(full_input, stream=True)
  full_response = "".join(chunk.text for chunk in response)
  headline_pattern = r"headline: (.+)"
  date_pattern = r"date: (.+)"
  author_pattern = r"author: (.+)"
  keywords_pattern = r"keywords: (.+)"
  source_pattern = r"source: (.+)"
  summary_pattern = r"summary: (.+)"

  state.article_title= re.search(headline_pattern, full_response).group(1)
  state.article_date = re.search(date_pattern, full_response).group(1)
  state.article_author = re.search(author_pattern, full_response).group(1)
  state.article_topic = re.search(keywords_pattern, full_response).group(1)
  state.article_source = re.search(source_pattern, full_response).group(1)
  state.article_summary = re.search(summary_pattern, full_response).group(1)
  state.chat_history = history
  
# Function for handling inputs into the generative AI model
def transform(input: str, history: list[mel.ChatMessage]):
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
        text_chunk = chunk.text
        yield chunk.text
        overall_sens_match = re.search(r'overall\s*sensationalism\s*:\s*(\d+(\.\d+)?)', text_chunk, re.IGNORECASE)
        overall_stance_match = re.search(r'overall\s*stance\s*:\s*(\d+(\.\d+)?)', text_chunk, re.IGNORECASE)
        if overall_sens_match:
            print('found_sens')
            state.overall_sens_score = float(overall_sens_match.group(1))
        if overall_stance_match:
            print('found_stance')
            state.overall_stance_score = float(overall_stance_match.group(1))
    state.chat_history = history
    # normal prompt get score
    # overall_sens_normal_prompt_match = re.search(r'normal\s*prompting\s*overall\s*sensationalism\s*:\s*(\d+(\.\d+)?)', text_chunk, re.IGNORECASE)
    # overall_stance_normal_prompt_match = re.search(r'normal\s*prompting\s*overall\s*stance\s*:\s*(\d+(\.\d+)?)', text_chunk, re.IGNORECASE)
    # if overall_sens_normal_prompt_match:
    #     state.overall_sens_normal_score = float(overall_sens_normal_prompt_match.group(1))
    # if overall_stance_normal_prompt_match:
    #     state.overall_stance_normal_score = float(overall_stance_normal_prompt_match.group(1))
    # cot prompt get score
    # overall_sens_cot_prompt_match = re.search(r'cot\s*prompting\s*overall\s*sensationalism\s*:\s*(\d+(\.\d+)?)', text_chunk, re.IGNORECASE)
    # overall_stance_cot_prompt_match = re.search(r'cot\s*prompting\s*overall\s*stance\s*:\s*(\d+(\.\d+)?)', text_chunk, re.IGNORECASE)
    # if overall_sens_cot_prompt_match:
    #     state.overall_sens_cot_score = float(overall_sens_normal_prompt_match.group(1))
    # if overall_stance_cot_prompt_match:
    #     state.overall_stance_cot_score = float(overall_stance_normal_prompt_match.group(1))
    print('checking for bug')
    # print(state.overall_sens_cot_score, state.overall_stance_cot_score)
    # fcot prompt get score
    # overall_sens_fcot_prompt_match = re.search(r'fcot\s*prompting\s*overall\s*sensationalism\s*:\s*(\d+(\.\d+)?)', text_chunk, re.IGNORECASE)
    # overall_stance_fcot_prompt_match = re.search(r'fcot\s*prompting\s*overall\s*stance\s*:\s*(\d+(\.\d+)?)', text_chunk, re.IGNORECASE)
    # if overall_sens_fcot_prompt_match:
    #     state.overall_sens_fcot_score = float(overall_sens_fcot_prompt_match.group(1))
    # if overall_stance_fcot_prompt_match:
    #     state.overall_stance_fcot_score = float(overall_stance_fcot_prompt_match.group(1))
    # print('checking for bug')
    # print(state.overall_sens_fcot_score, state.overall_stance_fcot_score)
    # state.normal_prompt_vs_fcot_prompt_log['normal_prompt'] = ["sensationalism: " + str(round(float(state.overall_sens_normal_score),2)),
    #                                                      "political_stance: " + str(round(float(state.overall_stance_normal_score),2))]
    # state.normal_prompt_vs_fcot_prompt_log['fcot_prompt'] = ["sensationalism: " + str(round(float(state.overall_sens_fcot_score),2)),
    #                                                          "political_stance: " + str(round(float(state.overall_stance_fcot_score), 2))]
    # state.normal_prompt_vs_fcot_prompt_log['cot_prompt'] = ["sensationalism: " + str(round(float(state.overall_sens_cot_score),2)),
    #                                                          "political_stance: " + str(round(float(state.overall_stance_cot_score), 2))]
    # print("####### prompt log ########")
    # print(state.normal_prompt_vs_fcot_prompt_log)
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
