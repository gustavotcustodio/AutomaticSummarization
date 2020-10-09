import os
import re


def get_word(word, dict_embedding):
    word = word.replace("…", "")
    if word not in dict_embedding:
        if word.capitalize in dict_embedding:
            return word.capitalize
        words = word.lower().split('-')
        embeds = " ".join([str(dict_embedding[word]) for word in words
                          if word in dict_embedding])
        return str(embeds)
    else:
        return str(dict_embedding[word])


dict_embedding = {}
with open('1-billion-word-language-modeling-benchmark-r13output.word2vec.vec',
          encoding='utf8') as f:
    print("Lendo palavras")
    for i in range(2, 559185):
        value = f.readline().split(' ')[0]
        dict_embedding[value] = i
    f.close()

txts_list = os.listdir("./sentences")

for txt in txts_list:
    with open('./sentences/%s' % txt, encoding='utf8') as f:
        list_of_embeddings = []
        sentences = f.read().split('\n')
        for sent in sentences:
            embeddings = [get_word(word, dict_embedding) for word in
                          sent.split(" ")]
            list_of_embeddings.append(" ".join(embeddings).strip())
        f.close()

    with open('./my_papers/%s' % txt, 'w', encoding='utf8') as f:
        output = re.sub(r'\s\s+', ' ', '\n'.join(list_of_embeddings))
        f.write(output)
        f.close()
    print("%s salvo com sucesso" % txt)
