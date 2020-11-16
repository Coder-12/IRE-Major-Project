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
# Here i summarized for cleaned_coa.txt and cleaned_nlp.txt

body=getdata(sys.argv[1])


# %%
# General Bert Summarizer
model = Summarizer()

# %%
result = model(body, min_length=60)

# %%
with open(sys.argv[2],'w') as fp:
    sentences=result.split('\n')
    for sent in sentences:
        fp.write(sent+'\n')
 fp.close()

# %%
# %%


