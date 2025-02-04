naive_realism_normal = [
    "Given the inputted article, rate it based on naive realism from a score of 1 to 10.\
Naive realism is determined by these three factors below:\
1. Perspective check: Evaluate if content assumes its perspective is the “only” correct one.\
2. Dissenting views check: Analyze if dissenting views are dismissed without consideration. \
3. Isolation check: Check if content aims to isolate readers from diverse perspectives. \
This should be done paragraph by paragraph. \
Then, combine them into an overall score for naive realism.\
Lastly please phrase overall score as 'Normal Prompting Overall naive realism: {score}'\
where score is a float and the phrase should be in plain text with no bolding or italics"]

naive_realism_cot = [
    "Given the inputted article, rate it based on naive realism from a score of 1 to 10.\
Naive realism is determined by these three factors below:\
1. Perspective check: Evaluate if content assumes its perspective is the “only” correct one.\
2. Dissenting views check: Analyze if dissenting views are dismissed without consideration. \
3. Isolation check: Check if content aims to isolate readers from diverse perspectives. \
Please explain your analysis and thought process for the rating \
This should be done paragraph by paragraph. \
Then, combine them into an overall score for naive realism.\
Lastly please phrase overall score as 'CoT Prompting Overall naive realism: {score}'\
where score is a float and the phrase should be in plain text with no bolding or italics"]

naive_realism_fcot = [
    "Given the inputted article, use 3 iterations. in each, determine what you missed in the previous iteration \
based on your evaluation, rate the aritcle based on naive realism from a score of 1 to 10. \
Naive realism is determined by these three factors below: \
1. Perspective check: Evaluate if content assumes its perspective is the “only” correct one.\
2. Dissenting views check: Analyze if dissenting views are dismissed without consideration. \
3. Isolation check: Check if content aims to isolate readers from diverse perspectives. \
Please explain your analysis and thought process for the rating \
This should be done paragraph by paragraph. \
Then, combine them into an overall score for naive realism.\
Then, give an overall score for naive realism based on the objective functions from a score of 1 to 10.\
Lastly please phrase overall score as 'Fcot Prompting Overall naive realism: {score}'\
where score is a float and the phrase should be in plain text with no bolding or italics"]