fcot_prompting_question = {
    # sensationalism question
    "Sensationalism": "Use 3 iterations. in each, determine what you missed in the previous iteration \
based on your evaluation of three objective functions defined as\
1. Sensationalism Detection: Identify instances of sensationalism in titles and main content.\
2. Emotion Analysis: Assess the writing style for excessive emotionality or exaggeration.\
3. Linguistic Database Comparison: Match linguistic features against databases of both \
trusted and untrusted sources to ascertain reliability.\
Think step by step and refine the current iteration to greater achieve each objective function.\
Then, give an overall score for sensationalism based on the objective functions from a score of 1 to 10.\
Lastly please phrase overall score as 'Fcot Prompting Overall sensationalism: {score}'\
where score is a float and the phrase should be in plain text with no bolding or italics",
    # political stance question
    "Political_stance": "Use 3 iterations. in each, determine what you missed in the previous iteration \
based on your evaluation of three objective functions defined as\
1. Perspective Analysis: Identify underlying perspectives on issues or events.\
2. Bias evaluation: Evaluate if the stance is consistently biased.\
3. verify facts: Compare the stance against verified facts.\
Think step by step and refine the current iteration to greater achieve each objective function.\
Then, give an overall score for political stance based on the objective functions from a score of 1 to 10.\
with 1 being extremely biased towards democratic and 10 being extremely biased towards conservative and 5 being neutral \
Lastly please phrase overall score as 'Fcot Prompting Overall Stance: {score}'\
where score is a float and the phrase should be in plain text with no bolding or italics"
}

# [

#     "Use 3 iterations. in each, determine what you missed in the previous iteration \
# based on your evaluation of three objective functions defined as\
# 1. Sensationalism Detection: Identify instances of sensationalism in titles and main content.\
# 2. Emotion Analysis: Assess the writing style for excessive emotionality or exaggeration.\
# 3. Linguistic Database Comparison: Match linguistic features against databases of both \
# trusted and untrusted sources to ascertain reliability.\
# Think step by step and refine the current iteration to greater achieve each objective function.\
# Then, give an overall score for sensationalism based on the objective functions from a score of 1 to 10.\
# Lastly please phrase overall score as 'Fcot Prompting Overall sensationalism: {score}'\
# where score is a float and the phrase should be in plain text with no bolding or italics",
# # political stance question
#     "Use 3 iterations. in each, determine what you missed in the previous iteration \
# based on your evaluation of three objective functions defined as\
# 1. Perspective Analysis: Identify underlying perspectives on issues or events.\
# 2. Bias evaluation: Evaluate if the stance is consistently biased.\
# 3. verify facts: Compare the stance against verified facts.\
# Think step by step and refine the current iteration to greater achieve each objective function.\
# Then, give an overall score for political stance based on the objective functions from a score of 1 to 10.\
# with 1 being extremely biased towards democratic and 10 being extremely biased towards conservative and 5 being neutral \
# Lastly please phrase overall score as 'Fcot Prompting Overall Stance: {score}'\
# where score is a float and the phrase should be in plain text with no bolding or italics"
# ]