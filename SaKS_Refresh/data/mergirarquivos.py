import os
# import random
import numpy as np

def score_sentence(sent_rouge):
    return float(sent_rouge.split(' - ')[0])


def save_multipleoracle(filename, summary_indexes, rouge_score, n_sents):
    str_indexes = " ".join([str(i) for i in summary_indexes])
    with open('./multipleoracle/%s' % filename, 'w') as f:
        f.write('%d\n' % n_sents)
        f.write('%s %s' % (str_indexes, rouge_score))
        f.close()


def generate_multipleoracle(all_files):
    for filename in all_files:
        # Ler valor do rouge
        rouge_file = open("./rouge/%s" % filename).read().split('\n\n')[1]
        sentences_rouge = rouge_file.split('\n')
        rouge_scores = np.array([score_sentence(sent_rouge) for sent_rouge in
                                sentences_rouge])
        labels = np.array(open("./labels/%s" % filename).read().split('\n'))
        summary_indexes = list(np.where(labels == '1')[0])
        rouge_mean = rouge_scores[summary_indexes].mean()
        save_multipleoracle(filename, summary_indexes, rouge_mean, len(labels))


def create_data(filelist, data_type, doc_or_label, dirname):
    with open('paperlist.%s.%s' % (data_type, doc_or_label), 'w') as f:
        for filename in filelist:
            content = open(os.path.join(dirname, filename)).read()
            f.write(filename + '\n')
            f.write(content.strip('\n') + '\n\n')
    f.close()

training_list = [
'zaharia2017.txt', 'yu2011.txt', 'bajo2012.txt', 'chang2011.txt', 'ponce2015.txt',
'rosasromero2016.txt', 'bae2012.txt', 'long2015.txt', 'clempner2017.txt', 'tollo2012.txt',
'khoshjavan2011.txt', 'ismail2011.txt', 'garci2013.txt', 'khashei2012.txt', 'duan2014.txt',
'les2013.txt', 'chou2014.txt', 'cardamone2013.txt', 'casabay2015.txt', 'rahman2011.txt',
'chiang2015.txt', 'yu2011a.txt', 'mostafa2011.txt', 'ferreira2012.txt', 'navarro2012.txt',
'ticknor2013.txt', 'chaaraoui2012.txt', 'tsui2014.txt', 'chow2013.txt',
'alvaradoiniesta2013.txt', 'chou2013.txt', 'garcaalonso2012.txt', 'asensio2014.txt',
'dahal2015.txt', 'chandwani2015.txt', 'jadhav2013.txt', 'hernndezdelolmo2012.txt',
'gurupur2015.txt', 'patel2011.txt', 'yanto2012.txt', 'ahn2012.txt', 'neokosmidis2013.txt',
'cavalcante2016.txt', 'deng2011.txt', 'xin2016.txt', 'silva2015.txt', 'nikoli2013.txt',
'buyukozkan2016.txt', 'garca2012a.txt', 'nunes2013.txt', 'crespo2013.txt', 'zheng2011.txt',
'deb2011.txt', 'li2014.txt', 'chou2014a.txt', 'clempner2016.txt', 'oliveira2013.txt',
'choudhury2010.txt', 'wang2013.txt', 'pandi2011.txt', 'laurentys2011.txt', 'abelln2017.txt',
'ahn2012a.txt', 'behera2012.txt', 'bielecki2013.txt', 'li2015.txt', 'patel2015.txt',
'segundo2017.txt', 'omoteso2012.txt', 'escario2015.txt', 'olawoyin2013.txt', 'wu2012.txt',
'soyguder2011.txt', 'titapiccolo2013.txt', 'gao2012.txt', 'adusumilli2013.txt',
'onieva2013.txt', 'affonso2015.txt', 'poggiolini2013.txt', 'lima2016.txt', 'cullar2011.txt',
'brady2017.txt', 'tan2011.txt', 'rmoreno2014.txt', 'esfahanipour2011.txt',
'kadadevaramath2012.txt', 'krishnasamy2014.txt', 'labib2011.txt', 'elsebakhy2011.txt',
'capozzoli2015.txt', 'marqus2012.txt', 'stavropoulos2013.txt', 'zelenkov2017.txt',
'zhang2013.txt', 'leite2014.txt', 'ramachandran2013.txt', 'maleszka2015.txt',
'chen2012.txt', 'gao2015.txt', 'wu2011a.txt', 'oreski2012.txt', 'bogaerd2011.txt',
'park2011.txt', 'vidoni2011.txt', 'castelli2013.txt', 'dias2013.txt', 'naranje2014.txt',
'marqus2012a.txt', 'leong2015.txt', 'falavigna2012.txt', 'montes2016.txt', 'araujo2014.txt',
'zhou2011.txt', 'ng2012.txt', 'coronato2014.txt', 'kele2011.txt', 'prezrodrguez2012.txt',
'yaghobi2011.txt', 'nascimento2013.txt', 'rouhi2015.txt', 'zhang2016a.txt',
'parente2015.txt', 'nikoli2013a.txt', 'duan2012.txt', 'gorriz2017.txt', 'ho2012.txt',
'das2014.txt', 'froz2017.txt', 'buche2011.txt', 'moncayomartnez2016.txt', 'paula2014.txt',
'brock2015.txt', 'leony2013.txt', 'cui2012.txt', 'boloncanedo2016.txt', 'tagluk2011.txt',
'wang2012.txt', 'samanta2011.txt', 'tasdemir2011.txt', 'sabzi2016.txt', 'villanueva2013.txt',
'hilaire2013.txt', 'parkinson2012.txt', 'mohdali2015.txt', 'nahar2012.txt',
'laalaoui2014.txt', 'liukkonen2012.txt', 'kovandi2016.txt', 'teodorovi2014.txt',
'wang2011.txt', 'hajizadeh2012.txt', 'duguleana2016.txt', 'alpar2015.txt', 'er2012.txt',
'garca2012.txt', 'tsai2011.txt', 'garcacrespo2011.txt', 'henriet2013.txt', 'lien2012.txt',
'rodrguezgonzlez2011.txt']

val_list = [
'pai2012.txt', 'yeow2014.txt', 'wu2011.txt', 'zhang2016.txt',
'mullen2013.txt', 'kizilkan2011.txt', 'gil2012.txt',
'saridakis2015.txt', 'mlakar2016.txt', 'moro2015.txt',
'krstanovi2016.txt', 'horng2011.txt', 'floresfernndez2012.txt',
'manahov2014.txt', 'subashini2016.txt', 'lpezcuadrado2012.txt',
'reis2014.txt', 'altnkaya2014.txt', 'ponce2014.txt',
'elsebakhy2012.txt',]

test_list = [
'herrero2011.txt', 'su2011.txt', 'hong2011.txt', 'aleksendri2012.txt',
'chou2011.txt', 'atici2011.txt', 'ardestani2014.txt', 'ribas2015.txt',
'yang2017.txt', 'ghiassi2012.txt', 'rafiei2011.txt', 'asiltrk2011.txt',
'bourguet2013.txt', 'pirovano2014.txt', 'oviedo2014.txt',
'andrades2013.txt', 'garg2014.txt', 'li2015a.txt', 'garcatorres2014.txt']

all_files = training_list + val_list + test_list

generate_multipleoracle(all_files)

create_data(all_files[:160], 'training', 'doc', './my_papers')
create_data(all_files[:160], 'training', 'label.singleoracle', './labels')
create_data(all_files[:160], 'training', 'label.multipleoracle',
            './multipleoracle')
create_data(all_files[160:180], 'validation', 'doc', './my_papers')
create_data(all_files[160:180], 'validation', 'label.singleoracle', './labels')
create_data(all_files[160:180], 'validation', 'label.multipleoracle',
           './multipleoracle')
create_data(all_files[180:], 'test', 'doc', './my_papers')
create_data(all_files[180:], 'test', 'label.singleoracle', './labels')
create_data(all_files[180:], 'test', 'label.multipleoracle', './multipleoracle'
            )
