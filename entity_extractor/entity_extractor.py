entities = []
tags = []
with open('ArmanPersoNERCorpus/test_fold1.txt') as file:
    for line in file.readlines():
        if line != '\n':
            entity, tag = line.split(' ')
            if tag != 'O\n':
                entities.append(entity)
                tags.append(tag)
with open('ArmanPersoNERCorpus/test_fold2.txt') as file:
    for line in file.readlines():
        if line != '\n':
            entity, tag = line.split(' ')
            if tag != 'O\n':
                entities.append(entity)
                tags.append(tag)

with open('ArmanPersoNERCorpus/test_fold3.txt') as file:
    for line in file.readlines():
        if line != '\n':
            entity, tag = line.split(' ')
            if tag != 'O\n':
                entities.append(entity)
                tags.append(tag)

with open('ArmanPersoNERCorpus/train_fold1.txt') as file:
    for line in file.readlines():
        if line != '\n':
            entity, tag = line.split(' ')
            if tag != 'O\n':
                entities.append(entity)
                tags.append(tag)
with open('ArmanPersoNERCorpus/train_fold2.txt') as file:
    for line in file.readlines():
        if line != '\n':
            entity, tag = line.split(' ')
            if tag != 'O\n':
                entities.append(entity)
                tags.append(tag)

with open('ArmanPersoNERCorpus/train_fold3.txt') as file:
    for line in file.readlines():
        if line != '\n':
            entity, tag = line.split(' ')
            if tag != 'O\n':
                entities.append(entity)
                tags.append(tag)
entity_unique = set(entities)
tag_type = set(tags)
with open('entities.txt', 'w') as file:
    for entity in entity_unique:
        file.write(entity + '\n')
print(entities)
