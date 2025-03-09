cot_prompting_question = {
    # sensationalism question
    "Sensationalism": "based on your evaluation of three objective functions defined as\
1. Sensationalism Detection: Identify instances of sensationalism in titles and main content.\
2. Emotion Analysis: Assess the writing style for excessive emotionality or exaggeration.\
3. Linguistic Database Comparison: Match linguistic features against databases of both \
trusted and untrusted sources to ascertain reliability.\
Provide explanation to your response to greater achieve each objective function.\
Then, give an overall score for sensationalism based on the objective functions from a score of 1 to 6.\
Lastly please phrase overall score as 'cot Prompting Overall sensationalism: {score}'\
where score is a float and the phrase should be in plain text with no bolding or italics",
    # political stance question
    "Political_stance": "based on your evaluation of three objective functions defined as\
1. Perspective Analysis: Identify underlying perspectives on issues or events.\
2. Bias evaluation: Evaluate if the stance is consistently biased.\
3. verify facts: Compare the stance against verified facts.\
Provide explanation to your response to greater achieve each objective function.\
Then, give an overall score for political stance based on the objective functions from a score of 1 to 6.\
with 1 being political neural towards democratic and 6 being extremely biased towards conservative or democratic \Lastly please phrase overall score as 'cot Prompting Overall Stance: {score}'\
where score is a float and the phrase should be in plain text with no bolding or italics"
}