import os
import re
from nltk.tokenize import sent_tokenize
from lxml import etree
from config import xmls_dir, papers_dir


def attach_sentences_to_xml(xml, sentences):
    for sentence in sentences:
        if not re.match(r'([1-9][0-9]?\.)+', sentence):
            sentence = sentence.replace('Download full-size image', '')
            xml_sentence = etree.Element('sentence')
            xml_sentence.text = re.sub(r'\s+|\n', ' ', sentence).strip()
            xml.append(xml_sentence)


def get_title(buffreader):
    title = etree.Element('title')
    title.text = buffreader.readline().strip()
    return title


def get_abstract(buffreader):
    """ Retorns the abstract to xml format. """
    xml_abstract = etree.Element('abstract')
    reading_abstract = False
    abstract_lines = []
    for line in buffreader:
        if line.strip() == 'Abstract':
            reading_abstract = True
        elif line.strip() == 'Keywords':
            break
        elif reading_abstract:
            abstract_lines.append(line.strip())
            sentences_abstract = sent_tokenize(' '.join(abstract_lines))
    # Tokeniza o abstract em frases e adiciona elas ao xml
    attach_sentences_to_xml(xml_abstract, sentences_abstract)
    return xml_abstract


def get_keywords(buffreader):
    """ Adiciona as palavras-chave do artigo para o xml."""
    xml_keywords = etree.Element('keywords')
    for line in buffreader:
        if not line.strip():
            break
        else:
            keyword = etree.Element('keyword')
            keyword.text = line.strip()
            xml_keywords.append(keyword)
    return xml_keywords


def get_article_sections(buffreader):
    list_xml_sections = []
    section_content = ""
    for line in buffreader:
        if re.match(r'^[1-9][0-9]?\.\s+.{3,}$', line):
            if list_xml_sections:  # Salvar informações da seção anterior
                attach_sentences_to_xml(list_xml_sections[-1], sent_tokenize(
                    section_content, 'english'))
            # Título de seção novo encontrado
            section_name = re.sub(r'^[1-9][0-9]?\.\s+', '', line).strip()
            list_xml_sections.append(
                etree.Element("section", attrib={"name": section_name}))
            section_content = ""
        else:
            section_content += line.replace('Fig.', 'Fig'
                                            ).replace('Eq.', 'Eq')
    attach_sentences_to_xml(list_xml_sections[-1],
                            sent_tokenize(section_content, 'english'))
    return list_xml_sections


paperlist = os.listdir(papers_dir)

print('Converting papers to xml format...')

for paper in paperlist:
    with open(os.path.join(papers_dir, paper)) as buffreader:
        xml_article = etree.Element('article')
        xml_article.append(get_title(buffreader))
        xml_article.append(get_abstract(buffreader))
        xml_article.append(get_keywords(buffreader))
        list_xml_sections = (get_article_sections(buffreader))
        for section in list_xml_sections:
            xml_article.append(section)

    with open(os.path.join(xmls_dir, paper.replace('.txt', '.xml')), 'w',
              encoding='utf-8') as xmlfile:
        xmlfile.write(etree.tostring(xml_article, encoding='utf-8',
                                     pretty_print=True).decode('utf-8'))
print('Papers successfully converted to xml.')
