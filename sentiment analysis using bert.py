#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformers requests beautifulsoup4 pandas numpy')


# In[7]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re


# In[9]:


tokenizer=AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model=AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


# In[10]:


tokens= tokenizer.encode('I hated this, absolutely the worst', return_tensors='pt')


# In[13]:


tokens[0]


# In[18]:


result=model(tokens)


# In[19]:


result


# In[72]:


r= requests.get('https://www.imdb.com/title/tt11737520/reviews?ref_=tt_urv')
soup= BeautifulSoup(r.text, 'html.parser')
regex= re.compile('.*text show-more__control.*')
results= soup.find_all('div',{'class':regex})
reviews=[result.text for result in results]


# In[73]:


reviews[0]


# In[52]:


import pandas as pd
import numpy as np


# In[77]:


df= pd.DataFrame(np.array(reviews), columns=['review'])


# In[89]:


def sentiment_score(review):
    tokens= tokenizer.encode(review, return_tensors='pt')
    result= model(tokens)
    return int(torch.argmax(result.logits))+1


# In[90]:


df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))


# In[92]:


df


# In[93]:


df['review'].iloc[15]


# In[ ]:




