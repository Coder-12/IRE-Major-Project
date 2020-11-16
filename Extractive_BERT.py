# %%
from summarizer import Summarizer
from pipelines import pipeline
import sys

# %%

# %%
!pip install -U spacy
# %%
# %%
def getdata(filename):
    data=""
    with open(filepath,'r') as fp:
        lines=fp.readlines()
        for line in lines:
           data+=line
    fp.close()
    return data
# %%
# Here it is summarized for cleaned_coa.txt and cleaned_nlp.txt

body=getdata(sys.argv[1])


# %%
# General Bert Summarizer
model = Summarizer()
result = model(body, min_length=60)
# %%
# Code To Generate Question From Text used code from https://github.com/patil-suraj/question_generation.git 
nlp = pipeline("question-generation")

# %%
sentences = result.split("\n")

for sent in sentences:
    if len(sent) == 0:
        continue
    try:
        res = nlp(sent)
    except ValueError:
        continue

    for x in res:
        q, a = x["question"], x["answer"]

        print(q, "\t", a)

# %%


# %%
# %%
# %%


