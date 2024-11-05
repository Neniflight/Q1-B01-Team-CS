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
  *  `liar plus dataset (test2, train2, val2)`: a dataframe that contains multiple columns of attibutes that is extracted from different text including the varying levels of misinformation labels
     *  `Column 1`: the ID of the statement ([ID].json).
     *  `Column 2`: the varying levels of misinformation
     *  `Column 3`: the title of the text
     *  `Column 4`: the subject(s) of the text (what categories the articles fall under)
     *  `Column 5`: the speaker/author.
     *  `Column 6`: the speaker's/author's job title.
     *  `Column 7`: the state the text was published/released/taken place
     *  `Column 8`: the party affiliation of the speaker/author
     *  `Columns 9-13`: the total credit history count, including the current statement.
        *  `column 9`: barely true counts.
        *  `column 10`: false counts.
        *  `column 11`: half true counts.
        *  `column 12`: mostly true counts.
        *  `column 13`: pants on fire counts.
     * `Column 14`: the context (venue/location/medium of the text).
     * `Column 15`: extracted justification that justifies the text's level of misinformation
  * `chunks.pkl`: List object that contains chunks from an article that contains misinformation
  * `speaker_reput_dict.pkl`: Dictionary object where the keys are speakers and the values are lists that contain the reputation of each speaker
* `model`: Contains our models
  * ` XGModel.sav`: XGBoosted Decision Tree that predicts veracity based on Naive Realism
  * ` social_cred_predAI.h5`: tensorflow Keras Neural Network that predicts social credibility score
  * ` social_cred_predAI.keras`: keras version of the saved tensorflow keras Neural Network to predict social credibility score
* `.gitignore:` Tell github which files not to track like env or pycache files.
* `.python-version`: Your python version should be 3.11.9 for app.py to work
* `Pred_AI_notebook`: Keras tensorflow Predictive AI and other attempted models on jupyternotebook. This is then cleaned and moved to social_credibility_predAI.py.
* `README.md`: What you are reading right now :)
* `app.py`: Python file that has all the functionality for the Mesop interface integrated with Generative AI and Predictive AI. This is where you **should start** to see how our main features work!
* `chunking.ipynb`: A Python Notebook that chunks an article. This is where `chunks.pkl` come from. 
* `chunking_and_chroma`: A Python Notebook that inputs these chunks into a vector database known as Chroma
* `environmental_mac.yml`: Used to download your environment to have all the packages to make this work. This env is for Mac devices
* `environmental_win.yml`: Used to download your environment to have all the packages to make this work. This env is for Window devices
* `questions.py`: Python file that contains a predefined list of questions to ask to the Generative AI.
* `social_credibility_predAI.py`: tensorflow keras neural network in python file to save and load on app.py

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
3. Create the environment based on your OS. Make sure you are using 3.11.9 or earlier because Google AI Studio does not support Python 3.13 and wheel does not support 3.12
```bash
conda env create -f environment_{respective_OS}.yml
```
4. Once done, you need to set up a ChromaDB database for the Google Gemini to base its responses off of. Run the `liar_plus_to_chroma.ipynb` in its entirety. Make sure you have Docker installed and running before you run this command:
```bash
docker run -d --rm --name chromadb -v ./chroma:/chroma/chroma -e IS_PERSISTENT=TRUE -e ANONYMIZED_TELEMETRY=TRUE -p 8000:8000 chromadb/chroma:0.5.13 
```
5. Start fighting against misinformation by starting the app. If you are experiencing issues with this, pip uninstall mesop and reinstall it again. On a different terminal, run the following:
```bash
mesop app.py
```
6. Once you are done messing around with the tool. Stop the docker container by running:
```bash
docker stop chromadb
```
---
# Members
* Calvin Nguyen: [Github](https://github.com/Neniflight), [Linkedin](https://www.linkedin.com/in/calvin-nguyen-data/)
* Samantha Lin: [Github](https://github.com/Samanthalin0918), [Linkdein](https://www.linkedin.com/in/samantha-lin-3bb601271/)
