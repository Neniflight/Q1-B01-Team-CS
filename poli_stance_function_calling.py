"""
This file is the functions writting for the Predictive AI part of our factuality factor Political Stance

There are a total of 3 functions in here: poli_stance_score, perspective_analyzer and verify_facts_analyzer
"""

# imports
import chromadb

"""
FUNCTION 1: poli_stance_score
    - This function is for calculating the final score for political stance by average all 3 microfactor's scores
    - Args: microfactor_1, microfactor_2, microfactor_3: float numbers
    - Output: float number of the final score for political stance
"""
# function to get the final score of poli_stance_score
def poli_stance_score(microfactor_1:float, microfactor_2:float, microfactor_3:float):
    """Averages the microfactors from a single factuality factor. This function should be used when combining into an overall score.
    
    Args:
        microfactor_1: A float value from 1 to 10 that represents the score of the microfactor of a factuality factor. 
        microfactor_2: A float value from 1 to 10 that represents the score of the microfactor of a factuality factor. 
        microfactor_3: A float value from 1 to 10 that represents the score of the microfactor of a factuality factor. 
    """
    score = (microfactor_1 + microfactor_2 + microfactor_3)/3
    print('poli_stance_score: ' + str(score))
    return score

# set up Chromadb and get collection
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
collection = chroma_client.get_or_create_collection(name="Misinformation")


"""
FUNCTION 2: perspective_analyzer
    - This function gets the top 10 most similar statemetns from our chromadb with party affiliation label, then 
        calculates the scores based on ther politial affiliations 
    - Args: text: str the user inputs
    - Output: float number of the score for this microfactor
"""
# MICRO FACTOR 1: Perspective Analyzer to analyze overall stance of the text
def perspective_analyzer(text:str):
    """Analyzes text's closeness to other text without considering truth or false statement, 
        scoring with the scale of 1 being extremely biased towards democratic and 10 being extremely 
        biased towards conservative and 5 being neurtral
    
    Args:
        text: A string value that represents the article we are grading on. 
    """
    # get the top 10 metadata from chromadb with political affiliation labels
    i = 1
    each_rst_party = []
    while len(each_rst_party) <= 10:
        results = collection.query(query_texts=[text], n_results=i)
        results = results['metadatas'][0][i-1]
        if 'party_affliation' in results.keys() and results['party_affliation'] != 'none':
            party = results['party_affliation']
            each_rst_party.append(party)
        i += 1

    # convert labels from party to numbers
    perspective_analyzer_poli_aff_score_list = []
    for label in each_rst_party:
        if label == "republican":
            perspective_analyzer_poli_aff_score_list.append(7)
        elif label == "ocean-state-tea-party-action":
            perspective_analyzer_poli_aff_score_list.append(10)
        elif label == "democrat":
            perspective_analyzer_poli_aff_score_list.append(3)
        elif label == "green":
            perspective_analyzer_poli_aff_score_list.append(2)
        elif label in ["democrat", "democratic-farmer-labor"]:
            perspective_analyzer_poli_aff_score_list.append(3)
        elif label == "liberal-party-canada":
            perspective_analyzer_poli_aff_score_list.append(4)
        else:
            perspective_analyzer_poli_aff_score_list.append(5)
    
    # calculating and returning the final score for this microFactor
    score = round(sum(perspective_analyzer_poli_aff_score_list) / 10, 2)
    print("perspective_analyzer: " + str(score))
    return score

# We will be evaluating MICRO FACTOR 2 with Gemini

"""
FUNCTION 3: verify_facts_analyzer
    - This function gets the top 10 most similar TRUE statemetns from our chromadb with party affiliation label, then 
        calculates the scores based on ther politial affiliations 
    - Args: text: str the user inputs
    - Output: float number of the score for this microfactor
"""
# MICRO FACTOR 3: verify facts analyzer to analyze overall stance of the text
def verify_facts_analyzer(text:str):
    """Analyzes text's closeness to other text with only TRUE statements, 
        scoring with the scale of 1 being extremely biased towards democratic and 10 being extremely 
        biased towards conservative and 5 being neurtral
    
    Args:
        text: A string value that represents the article we are grading on. 
    """
    # get the top 10 metadata from chromadb with political affiliation labels
    i = 1
    each_rst_party = []
    while len(each_rst_party) <= 10:
        results = collection.query(query_texts=[text], n_results=i, where = {"label": "true"})
        results = results['metadatas'][0][i-1]
        if 'party_affliation' in results.keys() and results['party_affliation'] != 'none':
            party = results['party_affliation']
            each_rst_party.append(party)
        i += 1

    # convert labels from party to numbers
    perspective_analyzer_poli_aff_score_list = []
    for label in each_rst_party:
        if label == "republican":
            perspective_analyzer_poli_aff_score_list.append(7)
        elif label == "ocean-state-tea-party-action":
            perspective_analyzer_poli_aff_score_list.append(10)
        elif label == "democrat":
            perspective_analyzer_poli_aff_score_list.append(3)
        elif label == "green":
            perspective_analyzer_poli_aff_score_list.append(2)
        elif label in ["democrat", "democratic-farmer-labor"]:
            perspective_analyzer_poli_aff_score_list.append(3)
        elif label == "liberal-party-canada":
            perspective_analyzer_poli_aff_score_list.append(4)
        else:
            perspective_analyzer_poli_aff_score_list.append(5)
    
    # calculating and returning the final score for this microFactor
    score = round(sum(perspective_analyzer_poli_aff_score_list) / 10, 2)
    print("verify_facts_analyzer: " + str(score))
    return score




perspective_analyzer("Shocking Discovery: Scientists Unveil a Secret That Will Change Humanity Forever!")
verify_facts_analyzer("Shocking Discovery: Scientists Unveil a Secret That Will Change Humanity Forever!")