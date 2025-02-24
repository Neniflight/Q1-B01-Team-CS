normal_prompting_question = {
    # sensationalism question
    "Sensationalism": "Given the inputted article rate it based on sensationalism from a score of 1 to 10.\
First, rate each chunk of the article based on the defined microfactors of sensationalism: \
1. Sensationalism Detection: Identify instances of sensationalism in titles and main content. \
2. Emotion Analysis: Assess the writing style for excessive emotionality or exaggeration.\
3. Linguistic Database Comparison: Match linguistic features against databases of both \
trusted and untrusted sources to ascertain reliability.\
Then, combine them into an overall score for sensationalism.\
Lastly please phrase overall score as 'Normal Prompting Overall sensationalism: {score}'\
where score is a float and the phrase should be in plain text with no bolding or italics",
    # political stance question
    "Political_stance": "Given the inputted article rate it based on political stance with 1 being extremely\
biased towards democratic and 10 being extremely biased towards conservative and 5 being neutral \
from a score of 1 to 10. First, rate each chunk of the article based on the defined \
microfactors of sensationalism: \
1. Perspective Analysis: Identify underlying perspectives on issues or events.\
2. Bias evaluation: Evaluate if the stance is consistently biased.\
3. verify facts: Compare the stance against verified facts.\
This should be done paragraph by paragraph. \
Then, combine them into an overall score for political stance.\
Lastly please phrase overall score as 'Normal Prompting Overall Stance: {score}'\
where score is a float and the phrase should be in plain text with no bolding or italics"
}


# [
# # sensationalism question
#     "Given the inputted article rate it based on sensationalism from a score of 1 to 10.\
# First, rate each chunk of the article based on the defined microfactors of sensationalism: \
# 1. Sensationalism Detection: Identify instances of sensationalism in titles and main content. \
# 2. Emotion Analysis: Assess the writing style for excessive emotionality or exaggeration.\
# 3. Linguistic Database Comparison: Match linguistic features against databases of both \
# trusted and untrusted sources to ascertain reliability.\
# Then, combine them into an overall score for sensationalism.\
# Lastly please phrase overall score as 'Normal Prompting Overall sensationalism: {score}'\
# where score is a float and the phrase should be in plain text with no bolding or italics",
# # political stance question
#     "Given the inputted article rate it based on political stance with 1 being extremely\
# biased towards democratic and 10 being extremely biased towards conservative and 5 being neutral \
# from a score of 1 to 10. First, rate each chunk of the article based on the defined \
# microfactors of sensationalism: \
# 1. Perspective Analysis: Identify underlying perspectives on issues or events.\
# 2. Bias evaluation: Evaluate if the stance is consistently biased.\
# 3. verify facts: Compare the stance against verified facts.\
# This should be done paragraph by paragraph. \
# Then, combine them into an overall score for political stance.\
# Lastly please phrase overall score as 'Normal Prompting Overall Stance: {score}'\
# where score is a float and the phrase should be in plain text with no bolding or italics"
# ]
