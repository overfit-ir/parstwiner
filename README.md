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
Please cite the following paper in your publication if you are using ParsBERT in your research:
```
@inproceedings{aghajani-etal-2021-parstwiner,
    title = "{P}ars{T}wi{NER}: A Corpus for Named Entity Recognition at Informal {P}ersian",
    author = "Aghajani, MohammadMahdi  and
      Badri, AliAkbar  and
      Beigy, Hamid",
    booktitle = "Proceedings of the Seventh Workshop on Noisy User-generated Text (W-NUT 2021)",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.wnut-1.16",
    pages = "131--136",
    abstract = "As a result of unstructured sentences and some misspellings and errors, finding named entities in a noisy environment such as social media takes much more effort. ParsTwiNER contains about 250k tokens, based on standard instructions like MUC-6 or CoNLL 2003, gathered from Persian Twitter. Using Cohen{'}s Kappa coefficient, the consistency of annotators is 0.95, a high score. In this study, we demonstrate that some state-of-the-art models degrade on these corpora, and trained a new model using parallel transfer learning based on the BERT architecture. Experimental results show that the model works well in informal Persian as well as in formal Persian.",
}
```
