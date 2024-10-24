import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
from dotenv import load_dotenv
import re
import mesop as me
import mesop.labs as mel
from dataclasses import field

from questions import predefined_questions

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

import mesop as me
import mesop.labs as mel

generation_config = {
    "max_output_tokens": 10000,
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
    system_instruction="You are trying to fight against misinformation by scoring different articles on their factuality factors. In your responses, do not use copyrighted material",
)

@me.stateclass
class State:
  file: me.UploadedFile
  overall_sens_score: float = 0.0
  overall_stance_score: float = 0.0
  predefined_questions: list = field(default_factory=lambda: predefined_questions)
  chat_history: list = field(default_factory=lambda: [])

def load(e: me.LoadEvent):
  me.set_theme_mode("system")

def handle_upload(event: me.UploadEvent):
  state = me.state(State)
  state.file = event.file

### THIS IS ERROR 
def ask_predefined_questions(event: me.ClickEvent):
    state = me.state(State)
    for question in state.predefined_questions:
        state.chat_history.append(mel.ChatMessage(role='user', content=question))
        responses = transform(question, state.chat_history)           
### 

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
        me.button(
           label="Rate Article on Factuality Factors",
           on_click=ask_predefined_questions,
           color="primary",
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
        me.text(f"Overall Democratic Stance: {state.overall_stance_score}", type="headline-5")
        me.progress_bar(mode="determinate", value=state.overall_stance_score*10, color='primary')

def transform(input: str, history: list[mel.ChatMessage]):
    state = me.state(State)
    print(history)
    chat_history = "\n".join(f"{message.role}: {message.content}" for message in history)
    full_input = f"{chat_history}\nuser: {input}"

    response = model.generate_content(full_input, stream=True)

    full_response_text = ""
    for chunk in response:
        text_chunk = chunk.text
        
        full_response_text += text_chunk + "\n"  # Collecting all text chunks
    # state.chat_history.append(mel.ChatMessage(role="assistant", content=full_response_text))
    overall_sens_match = re.search(r'overall\s*sensationalism\s*:\s*(\d+(\.\d+)?)', full_response_text, re.IGNORECASE)
    overall_stance_match = re.search(r'overall\s*stance\s*:\s*(\d+(\.\d+)?)', full_response_text, re.IGNORECASE)
    if overall_sens_match:
        state.overall_sens_score = float(overall_sens_match.group(1))
        print(state.overall_sens_score)
    if overall_stance_match:
        state.overall_stance_score = float(overall_stance_match.group(1))
    state.chat_history = history
    return full_response_text.splitlines()