import os
import numpy as np
import rouge

def score_sentence(sent_rouge, highlights, scorer):
    scores = scorer.get_scores(sent_rouge, highlights)
    avg_rouge = (scores['rouge-1']['f'] + scores['rouge-2']['f'] +
                 scores['rouge-l']['f']) / 3.0
    return avg_rouge

def save_rouge(filename, content_rouge_file, rouge_scores, sentences_rouge):
    # str_indexes = " ".join([str(i) for i in summary_indexes])
    new_scores_sents = ["%f - %s" % (score, sent)  for score, sent in
                        zip(rouge_scores, sentences_rouge)]
    content_rouge_file[1] = "\n".join(new_scores_sents)
    with open('./rouge2/%s' % filename, 'w') as f:
        f.write('\n\n'.join(content_rouge_file))
        f.close()
    print(filename + " salvo com sucesso.")

def generate_multioracle(all_files):
    scorer = rouge.Rouge(['rouge-n', 'rouge-l'], max_n=2, stemming=True)
    for filename in all_files:
        # Ler valor do rouge
        content_rouge_file = open("./rouge/%s" % filename).read().split('\n\n')
        highlights = open("./papers_highlights_rouge/%s" % filename
                          ).read().split('\n')
        sentences_rouge = content_rouge_file[1].split('\n')
        sentences_rouge = [sent.split(' - ')[1] for sent in sentences_rouge]
        rouge_scores = np.array([score_sentence(sent_rouge, highlights, scorer)
                                 for sent_rouge in sentences_rouge])
        save_rouge(filename, content_rouge_file, rouge_scores, sentences_rouge)
        # labels = np.array(open("./labels/%s" % filename).read().split('\n'))
        # summary_indexes = list(np.where(labels == '1')[0])
        # rouge_mean = rouge_scores[summary_indexes].mean()
        # save_multioracle(filename, summary_indexes, rouge_mean, len(labels))

if __name__ == "__main__":
    all_files = os.listdir('labels')
    generate_multioracle(all_files)
