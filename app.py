import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
from dotenv import load_dotenv, dotenv_values

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

import mesop as me
import mesop.labs as mel

import PyPDF2
from PyPDF2 import PdfReader

from dataclasses import field

generation_config = {
    "max_output_tokens": 8192,
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
    system_instruction="You are trying to fight against misinformation by scoring different articles on their factuality factors.",
)

@me.stateclass
class State:
  file: me.UploadedFile
  # chat_history: list = field(default_factory=lambda: [])


def load(e: me.LoadEvent):
  me.set_theme_mode("system")

@me.page(path="/chat", title="Gemini Misinformation ChatBot")
def page():
    state = me.state(State)
    with me.box(style=me.Style(padding=me.Padding.all(15), height="95%")):
        me.uploader(
          label="Upload PDF",
          accepted_file_types=[".pdf"],
          on_upload=handle_upload,
          type="flat",
          color="primary",
          style=me.Style(font_weight="bold"),
        )
        mel.chat(
            transform, 
            title="Gemini Misinformation Helper", 
            bot_user="Chanly", # Short for the Vietnamese word for Truth
        )
          

def transform(input: str, history: list[mel.ChatMessage]):
    state = me.state(State)
    chat_history="\n".join(message.content for message in history)
    chat_history += "\n" + pdf_to_text(state.file)
    full_input = f"{chat_history}\n{input}"
    response = model.generate_content(full_input, stream=True)
    for chunk in response:
        yield chunk.text

def handle_upload(event: me.UploadEvent):
    state = me.state(State)
    state.file = event.file
    # pdf_in_text = pdf_to_text(state.file)
    # print("hey")
    # state.chat_history.append(mel.ChatMessage(role="user", content=pdf_in_text))
    # print(state.chat_history)
    # print("hey2")
    # return state.chat_history

def pdf_to_text(user_pdf):
    reader = PdfReader(user_pdf)
    text = ""
    for i in range(len(reader.pages)):
      text_page = reader.pages[i]
      text += text_page.extract_text()
    return text
# citation: https://pypdf2.readthedocs.io/en/3.x/user/extract-text.html
# citation: https://www.geeksforgeeks.org/convert-pdf-to-txt-file-using-python/


# import google.generativeai as genai
# from google.generativeai.types import HarmCategory, HarmBlockThreshold
# import mesop as me
# import mesop.labs as mel
# import os

# genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# @me.page(path="/")
# def page():
#     mel.chat(transform, title="Gemini Misinformation Helper", bot_user="Chanly")

# generation_config = {
#     "max_output_tokens": 8192,
#     "response_mime_type": "text/plain",
#     "temperature": 1, # higher temp --> more risks the model takes with choices
#     "top_p": 0.95, # how many tokens are considered when producing outputs
#     "top_k": 40, # token is selected from 40 likely tokens
# }

# safety_settings = {
#   HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#   HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#   HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#   HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
# }

# model = genai.GenerativeModel(
#     model_name="gemini-1.5-pro-002",
#     generation_config=generation_config,
#     safety_settings=safety_settings,
#     system_instruction="You are trying to fight against misinformation by scoring different articles on their factuality factors.",
# )

# def transform(input: str, history: list[mel.ChatMessage]):
#     chat_history="\n".join(message.content for message in history)
#     full_input = f"{chat_history}\n{input}"
#     response = model.generate_content(full_input, stream=True)
#     for chunk in response:
#         yield chunk.text