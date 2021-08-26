<h1>تعیین اسامی خاص در توییتر فارسی</h1>
<p>  
In the preparation of the present data, MUC and CoNLL methods have been used. 
Entity mentions are annotated as continuous, non-overlapping spans of text that are assigned
exactly one type from the following categories:
- PER: person
- LOC: location, 
- ORG: organizations
- EVE: events
- POG: political groups and historical dynasties
- NAT: nationalities and ethnicities.
The <a href="https://github.com/ICTRC/Parsivar">Parsivar</a> tool is used to tokenize words. Before tokenizing the words, all the emojis and the links and the usernames, and the hashtag sign (#) have been removed. Then tweets were normalized using the Parsivar tool, and then some symbols including _ , + , ] , [  were removed, the details of which are given in the <a href="https://github.com/overfit-ir/persian-twitter-ner/blob/master/tokenizer.ipynb">tokenizer</a> file can be viewed. The human agent then reviews the tokenizing operation using the following points.
</p>

<p>
1. The adjectives that are inside the words are labeled as entities, such as "خلیج همیشگی فارس" (the forever Persian Gulf), where the word "همیشگی" (forever) is labeled LOC.
</p>

<p>
2. The first indicators of nouns are not labeled as existence, for example, in the "دکتر ظریف " (Dr. Zarif), the word "دکتر" (Dr.) should not be labeled as PER. The first index of words is labeled only if the deletion of that index causes the remaining words not to mean specific names. For example, in the word "Imam Zaman", the word "امام" (Imam) should also be labeled PER.
</p>

<p>
3. Historical dynasties such as "هخامنشیان" (the Achaemenids) or "قاجار" (the Qajars) are labeled POG.
</p>

<p>
4. If the pronouns or suffixes were attached to the entities, the whole word would be labeled, for example, the word "ایرانی‌ام" (I am Iranian) would be labeled NAT.
</p>

<p>
5. In preprocessing phase, all words in hashtags have been separated with spaces. for example "ایران_عزیز#" (#dear_iran) changed to "ایران عزیز" (dear Iran)
</p>

<p>
6. The universities and the schools and the prisons are labeled ORG.
</p>

<p>
7. Words  such as "یزدی" (Yazdi), "اصفهانی" (Isfahani), etc that show the belonging to a city or locality are also labeled NAT.
</p>

<p>
8. The plural sign such as "ها" (plural sign "s") etc. are considered part of the word.
</p>

<p>
۹. رشته توییت ها یا آخرین جمله ناقص آنها حذف شده و یا اگر نقص زیادی داشتند اولین جمله توییت بعدی به آنها اضافه شده است.
</p>

<p>
۱۰. در بین توییت‌ها برخی شامل اشعار هستند که همانند بقیه توییت‌ها با آن‌ها برخورد شده است.
</p>

<p>
۱۱. علایم سجاوندی هر یک به تنهایی به عنوان یک توکن در نظر گرفته می‌شوند مگر در حالاتی که چند توکن معنی یک شکلک بدهد. مثلا علامت :( به عنوان یک توکن در نظر گرفته شده است.
</p>

<p>
۱۲.  کلمات انگلیسی اگر در آخر توییت باشند مثلا به عنوان هشتگ یا غیره پاک می‌شوند اما چنانچه در وسط متن باشند بدون تغییر باقی می‌مانند و چنانچه به موجودیت خاص اشاره کنند برچسب هم می‌گیرند.
</p>

<p>
۱۳. افعالی مانند "بوده است" یا "شده بود" به صورت کلمات جدا در نظر گرفته شده‌اند.
</p>

<p>
۱۴. لفظ‌های "تر" و "ترین" هم جزیی از کلمه متبوع خودشان لحاظ شده‌اند.
</p>
