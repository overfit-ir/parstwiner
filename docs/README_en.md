<h1>Persian Twitter NER (ParsTwiner)</h1>

<p> 
This document details the guidelines for the annotation of the ParsTwiner corpus, a broad-coverage corpus for Persian named entity recognition.
The annotation follows the guidelines of the MUC and CoNLL corpuses. 

Entity mentions are annotated as continuous, non-overlapping spans of text that are assigned exactly one type from the following categories:
  
- PER: Person
- LOC: Location
- ORG: Organizations
- EVE: Events
- POG: Political groups and historical dynasties
- NAT: Nationalities and ethnicities.
  
Before tokenizing, all the emojis, links, usernames, and hashtag sign (#) are removed. The <a href="https://github.com/ICTRC/Parsivar">Parsivar</a> tool is used to tokenize and normalize, and then some symbols including _ , + , ] , [  are removed. The details of preprocessing and tokenizing are in the <a href="https://github.com/overfit-ir/persian-twitter-ner/blob/master/tokenizer.ipynb">tokenizer</a> file. At the end, The human agents review the tokenizing operation using the following points:
</p>

<p>
1. The adjectives that are between the two words are labeled as entities, such as in the phrase "خلیج همیشگی فارس" (the forever Persian Gulf), where the word "همیشگی" (forever) is between two location entities "خلیج" (Gulf) and "فارس" (Persian) so that is labeled LOC.
</p>

<p>
2. The pre-eminent of noun are not labeled as entity, for example, in the phrase "دکتر ظریف " (Dr. Zarif), the word "دکتر" (Dr.) should not be labeled as PER. The pre-eminent of words is labeled only if the deletion of that pre-eminent causes the remaining words not to mean entites. For example, in the phrase "امام زمان" (Imam Zaman), the word "امام" (Imam) should also be labeled as PER beacuse "امام زمان" (means Imam of time) refers to a specific person but the just the word "زمان" (zaman) lonely means "time" and is not a entity.
</p>

<p>
3. Historical dynasties such as "هخامنشیان" (the Achaemenids) and "قاجار" (the Qajars) are also labeled POG.
</p>

<p>
4. If the pronouns or suffixes were attached to the entities, the whole word would be labeled, for example, the word "ایرانی‌ام" (I am Iranian) would be labeled NAT.
</p>

<p>
5. In preprocessing phase, all words in hashtags are separated with spaces. for example "ایران_عزیز#" (#dear_iran) changes to "ایران عزیز" (dear Iran)
</p>

<p>
6. The universities, the schools, and the prisons are labeled ORG.
</p>

<p>
7. Words such as "یزدی" (Yazdi), "اصفهانی" (Isfahani) that show the belonging to a city or locality are also labeled NAT.
</p>

<p>
8. The plural sign such as "ها" (means plural sign "s") and etc. are considered as part of the word.
</p>

<p>
9. For the threads of tweets, their last incomplete sentence is deleted, or if it is needed, the first sentence of the next tweet is added to them.
</p>

<p>
10. Some of the tweets include poems and they have been considered as any other tweet.
</p>

<p>
11.Each of The punctuation marks is considered as a single token, except in cases that multiple tokens mean an emoji, for example, the symbol :) is considered as one token.
</p>

<p>
12. English words are deleted if they are at the end of a tweet, for example as a hashtag, but if they are in the middle of the text, they remain unchanged, and if they refer to a specific entity, they are also labeled.
</p>

<p>
13. Verbs such as "بوده است" (had p.p.) or "شده بود" (have p.p.) are considered as separate words.
</p>

<p>
14. The suffixes "تر" (-er like in "bigger") and "ترین" (-est like in "the biggest") are also part of their own word.
</p>
