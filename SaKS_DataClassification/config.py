import nltk
import yaml

nltk.download('punkt')
with open('config.yml') as yaml_file:
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    xmls_dir = params['xmls_dir']
    papers_dir = params['papers_dir']
    highlights_dir = params['highlights_dir']
    preprocessed_dir = params['preprocessed_dir']
