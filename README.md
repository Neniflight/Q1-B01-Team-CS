# Q1-B01-Team-CS #
<p>Description: This project was done as part of the DSC 180A Capstone with <b>Dr. Arsanjani</b>. We are fighting against misinformation by utilizing factuality factors and veracity vectors. Then we will be using predictive models and large language models to inform us on whether a certain article is misinformation or not. There are varying levels of misinformation labels from:</p>

* pants-on-fire
* mostly-false
* false
* half-true
* mostly-true
* true
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
4. Once done, you need to set up a ChromaDB database for the Google Gemini to base its responses off of. Run the `liar_plus_to_chroma.ipynb` in its entirety. Make sure you have Docker installed.
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
* Samantha Lin
