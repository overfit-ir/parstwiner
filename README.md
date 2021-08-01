# Persian Twitter NER (ParsTwiner)

An open, broad-coverage corpus for informal Persian named entity recognition collected from Twitter.

![Example of ParsTwiNER corpus annotation](https://github.com/overfit-ir/persian-twitter-ner/raw/master/docs/example.png)

## Releases

Version 1.0:

* zip package: [ParsTwiNER-v1.0.zip](https://github.com/overfit-ir/persian-twitter-ner/releases/download/v1.0.0/ParsTwiNER.zip)

Recommended. This is the first complete, stable release of the corpus and the version used in our experiments with the data.

## Quickstart

A version of the corpus data is found in CoNLL-like format in the following files:

* `twitter_data/persian-ner-twitter-data/train.txt`: training data
* `twitter_data/persian-ner-twitter-data/dev.txt`: development data
* `twitter_data/persian-ner-twitter-data/test.txt`: test data

These files are in a simple two-column tab-separated format with IOB2 tags:

```
این	O
تاج‌الدین	B-PER
همونه	O
که	O
دخترش	O
دور	O
قبل	O
نماینده	O
اصفهان	B-LOC
بود	O
```

The corpus annotation marks mentions of person (`PER`), organization (`ORG`), location (`LOC`), nations (`NAT`), political groups (`POG`), and event (`EVENT`) names.


## Guidelines

The [ParsTwiNER annotations instructions](https://github.com/overfit-ir/persian-twitter-ner/blob/master/docs/README.md) are available in MD format.

## Reference

