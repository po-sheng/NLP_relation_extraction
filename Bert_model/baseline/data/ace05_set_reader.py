from multi_doc_analyzer.structure.structure import *
import sys
sys.path.append("/work/relation_extraction/Bert_model/baseline/data")
from ace05_set_parser import parse_source_english, parse_source_chinese
from stanfordcorenlp import StanfordCoreNLP

class ACE05Reader:
	"""
	Read and parse ACE-2005 source file (*.sgm) and annotation file (*.xml).
	
	"""
	def __init__(self, lang: str, nlp: StanfordCoreNLP=None):
		"""
		:param lang: the language which is read_file {en:english, zh:chinese}
		"""
		self.nlp = nlp
		self.lang = lang
	
	def read(self, fp) -> Dict[str, Document]:
		"""
		:param fp: file path of ACE-2005 (LDC2006T06), e.g., '~/work/data/LDC2006T06/'
		:return: Dict[documentID, Document]
		"""
		if self.lang == 'en':
			doc_dicts = parse_source_english(fp, self.nlp)
			return doc_dicts
		elif self.lang == 'zh':
			doc_dicts = parse_source_chinese(fp)
			return doc_dicts


if __name__=='__main__':
	nlp = StanfordCoreNLP()

	# english
	reader = ACE05Reader('en')
	doc_dicts = reader.read('/media/moju/data/work/resource/data/LDC2006T06/')
	print("number of English docs:{}".format(len(doc_dicts)))
	
	# chinese
	reader = ACE05Reader('zh')
	doc_dicts = reader.read('/media/moju/data/work/resource/data/LDC2006T06/')
	print("number of Chinese docs:{}".format(len(doc_dicts)))
