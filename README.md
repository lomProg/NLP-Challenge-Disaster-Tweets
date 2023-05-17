# NLP Challenge Disaster Tweets
Twitter has become an important communication channel in times of emergency. The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter 
(i.e. disaster relief organizations and news agencies). 

# Development
The project is divided into three very specific phases:
 - Pre-processing,
 - Word embedding,
 - Text Classification.
## Pre-Processing
In any NLP task, cleaning raw text data is an fundamental step. We will be following the next steps to clean the raw tweets in out data:
1. Removal of URLS, hashtags and usernames
2. Conversion of emojis and emoticons in a significant word (e.g. :) -> *positive*)
3. Decontractions of slangs and english contracted forms (idk-> *i don't know*, I'm -> *I am*)
4. Text to lowercase 
5. Removal of digits, stopwords, puntuaction and white spaces
6. Removal of particular combinations of symbols

In addition we performed the **lemmatization** and **stemming** of the cleaned text.


