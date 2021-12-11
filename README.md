# InstaHireAI

## Chatbot
What makes our application InstaHireAI "THE INSTANT" hiring system is the customer centric approach that we took, especially keeping generation X in mind where they need everything instantly. Apart from the traditional form-based user input we have the provision for fast, easy, and fun chatbot inputs where the use can interact with our chatbots and get all the details like education, experience, skillsets filled in instantly and with ease.	 

To implement such an interactive chatbot we have used NLP and RiveScript, which is a text-based scripting language.

## Resume Segregation is an import feature for the hiring companies:

Now that we have all the details and resumes of say hundreds of candidates, from a recruiters point of view the challenge would be how to select the top candidate for the given job opening. 
The recruiters can have hiring teams to review resumes manually to find the best candidate or outsource the manually screening of the resumes to a hiring agency. Both are slow and takes a lot of human effort.
 Is there a way to automatically weeding out resumes that aren’t a fit for our specific role?
Our answer is YES!!! Our InstaHire’s “ResumeAnalyzer” does that exactly. It ranks resumes based on the job requirement. 

We have used pdfminer to extract the words from the text and used Spacy phrasematcter.
Applying the matcher to a document of words gives us access to the matched tokens in context.
And, Since spaCy is used for processing both the patterns and the text to be matched, we won’t have to worry about specific tokenization and won’t have to write a complex token pattern covering the exact tokenization of the term.
Once we have the matched tokens/keywords it’s given a score. The candidates are ranked based on scores.
So, the companies can use this tool to call in the best candidates for further assessments.
This saves time from manually reviewing the resumes and also removes the cost for the companies in hiring an agency and thus results in a faster candidate pipeline. Making instaHireAI truly an instant hiring platform. 

## Job Recommendation is a feature for the candidates looking for jobs

As most of the tech jobs requires an interdisciplinary skill, sometimes candidates are confused which job roles that best suits their skillsets. So, we are providing them with job recommendations.
It can improve their chances of landing a specific job.
To solve this problem, we are using deep learning just not traditional machine learning. Because, NLP task like automatic summarization and topic extraction requires to sort the corpus of text into different feature groups like bigrams and trigrams, the problem with that is if there are too few features the model will have a hard time making a useful prediction if there are too many the model can be slow and overfit. We don’t really know what feature representation is best for a given task and if someone does it still relies on  a human being in the loop,  ideally we just want to be able to give a raw dataset to a model and have it produce an insight without any human help. 
Deep Learning allows these features to be learnt automatically.
So how we are doing it?
We collected job description snippets using the indeed job search API, it enables access to shortened job summaries given a keyword. The keywords we have used were 25 different IT job categories like Data Scientist, Data Engineer, Data Analyst and these tiles were used as labels for training. For testing the model we used sample resumes. 

Turning words to vectors using word to vec. Then feeding those vectors in our Keras RNN model to learning the mapping between those word vectors and the job category label.
Given a new resume it’s converted it into a vector then feed into the model and we get the percentage likelihood that it belongs to a certain job category is the output. If that percentage is over a certain threshold that we predefined than it’s valid for our job. 

## Job Recommendation (Evaluation Metrics) 

The model did pretty well. The train accuracy was about 95% and 65% for testing accuracy.
We have used Deep Learning because rather than engineering features, we can let deep learning, learn relevant features and we can focus on tuning the model architecture for better performance. 
In most hiring platforms the hiring pipeline currently requires lots of unnecessary, costly efforts. We have leveraged the power of AI and big data to make the process more efficient for both candidates and for recruiters.
