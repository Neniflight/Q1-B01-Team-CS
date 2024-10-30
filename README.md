# Q1-B01-Team-CS #
<p>Description: This project was done as part of the DSC 180A Capstone with <b>Dr. Arsanjani</b>. We are fighting against misinformation by utilizing factuality factors and veracity vectors. Then we will be using predictive models and large language models to inform us on whether a certain article is misinformation or not. There are varying levels of misinformation labels from:</p>

* pants-on-fire
* mostly-false
* false
* half-true
* mostly-true
* true

---
# File Description #
Each file on this repo has its own purpose! We will explain what each file does below
* `data`: Contains our dataset, liar plus, and other data needed to work notebooks.
  * `chunks.pkl`: List object that contains chunks from an article that contains misinformation
  * `speaker_reput_dict.pkl`: Dictionary object where the keys are speakers and the values are lists that contain the reputation of each speaker
* `model`: Contains pickles of our models
  * ` XGModel.sav`: XGBoosted Decision Tree that predicts veracity based on Naive Realism
* `.gitignore:` Tell github which files not to track like env or pycache files.
* `.python-version`: Your python version should be 3.12.7 for this code to work
* `README.md`: What you are reading right now :)
* `app.py`: Python file that has all the functionality for the Mesop interface integrated with Generative AI and Predictive AI. This is where you **should start** to see how our main features work!
* `chunking.ipynb`: A Python Notebook that chunks an article. This is where `chunks.pkl` come from. 
* `chunking_and_chroma`: A Python Notebook that inputs these chunks into a vector database known as Chroma
* `environmental_mac.yml`: Used to download your environment to have all the packages to make this work. This env is for Mac devices
* `environmental_win.yml`: Used to download your environment to have all the packages to make this work. This env is for Window devices
* `questions.py`: Python file that contains a predefined list of questions to ask to the Generative AI. 

---
# How to Get Started
1. Clone this repo with the following code:
```bash
git clone {github_repo_link}
```
2. Create an .env file in the root directory and paste your [Google AI Studio API key](https://aistudio.google.com/apikey) inside of it:
```bash
GEMINI_API_KEY="{API_KEY}"
```
3. Create the environment based on your OS. Make sure you are using 3.12.7 or earlier because Google AI Studio does not support Python 3.13
```bash
conda env create -f environment_{respective_OS}.yml
```
4. Start fighting against misinformation by starting the app
```bash
mesop app.py
```
---
# Members
* Calvin Nguyen: [Github](https://github.com/Neniflight), [Linkedin](https://www.linkedin.com/in/calvin-nguyen-data/)
* Samantha Lin
