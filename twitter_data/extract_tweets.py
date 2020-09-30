import json
import re

link_pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
id_pattern = re.compile("@[A-Za-z0-9-_]+")
emoji_pattern = re.compile(pattern="["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00010000-\U0010ffff"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)

# %%
with open('twitter_data/result.json') as f:
    crawled_data = json.load(f)

# %%
result = {}
with open('twitter_data/extracted_data.txt', 'w') as f:
    for entity in crawled_data:
        for keyword in entity.keys():
            keyword_result = {}
            list_text = []
            for key, tweet in entity[keyword]['globalObjects']['tweets'].items():
                text = tweet['full_text']
                lang = tweet['lang']

                keyword_result.update(
                    {key: {'full_text': text,
                           'lang': lang,
                           }
                     }
                )
                if lang == 'fa':
                    removed_link = link_pattern.sub('', text)
                    removed_username = id_pattern.sub('', removed_link)
                    removed_hashtags = removed_username.replace('#', '')
                    removed_emoji = emoji_pattern.sub('', removed_hashtags)
                    if removed_emoji.find(keyword) != -1:
                        list_text.append(removed_emoji)

            if len(set(list_text)) > 0:
                f.write('\n\n**************\n\n'.join(set(list_text[:2])))
                f.write('\n\n**************\n\n')
            result.update({keyword: keyword_result})
