import json
import glob
import os
from tqdm import tqdm

from multi_doc_analyzer.corpus_reader.ace2005_reader.apf_xml import ApfRelation, ApfEntity
from multi_doc_analyzer.corpus_reader.ace2005_reader.apf_xml_parser import parse_apf_docs
from multi_doc_analyzer.corpus_reader.ace2005_reader.sgm_parser import SgmDoc, parse_sgms_english, parse_sgms_chinese
from multi_doc_analyzer.structure.structure import *
from typing import *
from stanfordcorenlp import StanfordCoreNLP

DEBUG = 0


def is_in_sentence(index, sentence_start, sentence_end):
	if index < sentence_start:
		return False
	elif index > sentence_end:
		return False
	else:
		return True


def clean_string(string):
	out_string = ''
	for t in string:
		if not (t == "\n" or t == " "):
			out_string += t
	return out_string


def index_clean(sent_start, sent_string, arg_index):
	arg_index -= sent_start
	delete_n = 0
	for t_index, t in enumerate(sent_string):
		if t == "\n" or t == " ":
			assert (arg_index != t_index)
			if arg_index > t_index:
				delete_n += 1
	return arg_index - delete_n


def arg_to_word_idx(mentionArg_dic, mySentence):
	out_start = None
	out_end = None
	for idx, word in enumerate(mySentence.words):
		if word.start == mentionArg_dic['start']:
			out_start = idx
		if word.end == mentionArg_dic['end']:
			out_end = idx
	
	return out_start, out_end


def preserve_relation_example(relation, rMention, out_relation_list, sentence_index, sentence, nlp, props,
                              word_seg_error):
	out_relation_mention = {}
	mentionArg1_dic = {}
	mentionArg2_dic = {}
	
	out_relation_mention['relationID'] = relation.id
	out_relation_mention['relationType'] = relation.type
	out_relation_mention['relationSubType'] = relation.subtype
	out_relation_mention['id'] = rMention.id
	
	out_relation_mention['start'] = index_clean(sentence.start, sentence.string, rMention.extent.start)
	out_relation_mention['end'] = index_clean(sentence.start, sentence.string, rMention.extent.end)
	out_relation_mention['extent'] = clean_string(rMention.extent.text)
	
	mentionArg1_dic['start'] = index_clean(sentence.start, sentence.string, rMention.arg1.extent.start)
	mentionArg1_dic['end'] = index_clean(sentence.start, sentence.string, rMention.arg1.extent.end)
	mentionArg1_dic['extent'] = clean_string(rMention.arg1.extent.text)
	
	mentionArg2_dic['start'] = index_clean(sentence.start, sentence.string, rMention.arg2.extent.start)
	mentionArg2_dic['end'] = index_clean(sentence.start, sentence.string, rMention.arg2.extent.end)
	mentionArg2_dic['extent'] = clean_string(rMention.arg2.extent.text)
	
	out_relation_mention['sentence_index'] = sentence_index
	
	out_relation_mention['chars'] = []
	for t in sentence.string:
		if not (t == "\n" or t == " "):
			out_relation_mention['chars'].append(t)
	
	sentence_string = ''
	for t in out_relation_mention['chars']:
		sentence_string += t
	
	annotation = nlp.annotate(sentence_string, props)
	annotation = json.loads(annotation)
	
	for t in annotation['tokens']:
		sentence.to_words(t)
	
	mentionArg1_dic['start'], mentionArg1_dic['end'] = arg_to_word_idx(mentionArg1_dic, sentence)
	mentionArg2_dic['start'], mentionArg2_dic['end'] = arg_to_word_idx(mentionArg2_dic, sentence)
	
	out_relation_mention['Tokens'] = [word.string for word in sentence.words]
	out_relation_mention['Sentence'] = sentence_string
	out_relation_mention['sentence_length'] = len(out_relation_mention['chars'])
	out_relation_mention['mentionArg1'] = mentionArg1_dic
	out_relation_mention['mentionArg2'] = mentionArg2_dic
	
	if not (mentionArg1_dic['start'] == None or mentionArg1_dic['end'] == None or mentionArg2_dic['start'] == None or
	        mentionArg2_dic['end'] == None):
		out_relation_list.append(out_relation_mention)
	else:
		word_seg_error += 1
	return word_seg_error


def merge_sgm_apf(sgm_dicts: Dict[str, SgmDoc], doc2entities: Dict[str, Dict[str, ApfEntity]],
				  doc2relations: Dict[str, Dict[str, ApfRelation]]) -> Dict[str, Document]:
	if DEBUG == 1:
		em_cross_count = 0
		r_cross_count = 0
		args_cross_count = 0

	doc_dicts = {}
	for docID, sgm_doc in tqdm(sgm_dicts.items()):
		sentences = []
		for sentence_index, sgm_s in enumerate(sgm_doc.sentences):

			# entity mention
			e_dicts = {}
			e_mentions = []
			for e_id, apf_e in doc2entities[docID].items():
				entity_id = apf_e.id

				for apf_m in apf_e.mentions:
					char_b = apf_m.head.start
					char_e = apf_m.head.end
					if sgm_s.start <= char_b <= sgm_s.end:
						if not (sgm_s.start <= char_e <= sgm_s.end):  # e_mention cross sentence
							if DEBUG == 1:
								em_cross_count += 1
						else:
							e_mention = EntityMention(id=apf_m.id, tokens=[], type=apf_e.type,
													  char_b=char_b - sgm_s.start, char_e=char_e - sgm_s.start)
							e_mention.text = apf_m.head.text
							e_mentions.append(e_mention)
							e_dicts[apf_m.id] = e_mention
			# relation mention
			r_mentions = []
			for relation_id, apf_r in doc2relations[docID].items():

				for apf_rm in apf_r.mentions:
					r_mention_start = apf_rm.extent.start
					r_mention_end = apf_rm.extent.end

					if sgm_s.start <= r_mention_start <= sgm_s.end:
						if not (sgm_s.start <= r_mention_end <= sgm_s.end):  # relation extent cross sentence
							if DEBUG == 1:
								r_cross_count += 1
								print(sgm_s.string)
								print(docID)
								print(apf_rm.extent.text)
								print("arg1:{}".format(apf_rm.arg1.extent.text))
								print("arg2:{}".format(apf_rm.arg2.extent.text))
						if not (is_in_sentence(apf_rm.arg1.extent.start, sgm_s.start, sgm_s.end) and
								is_in_sentence(apf_rm.arg1.extent.end, sgm_s.start, sgm_s.end) and
								is_in_sentence(apf_rm.arg2.extent.start, sgm_s.start, sgm_s.end) and
								is_in_sentence(apf_rm.arg2.extent.end, sgm_s.start, sgm_s.end)):
							if DEBUG == 1:
								# args cross sentence
								args_cross_count += 1
						else:
							r_mentions.append(RelationMention(id=apf_rm.id, type=apf_r.type,
															  arg1=e_dicts[apf_rm.arg1.id],
															  arg2=e_dicts[apf_rm.arg2.id]))

			s = Sentence(id=sentence_index, tokens=sgm_s.tokens, string=sgm_s.string, char_b=sgm_s.start, char_e=sgm_s.end)
			s.entity_mentions = e_mentions
			s.relation_mentions = r_mentions
			sentences.append(s)
		doc_dicts[docID] = Document(id=len(doc_dicts), sentences=sentences)

		if DEBUG == 1:
			print('em_cross_count:{} r_cross_count:{} args_cross_count:{}'.
				  format(em_cross_count, r_cross_count, args_cross_count))

	return doc_dicts


def clean_docs_english(doc_dicts):
	for doc_id, doc in doc_dicts.items():
		for sentence in doc.sentences:
			space_string = ''
			for char in sentence.string:
				if char in ['\n', '\t', '\r']:
					space_string += ' '
				else:
					space_string += char

			cleaned_string = ''
			last_char = ''
			del_ids: List[int] = []
			for char_id, char in enumerate(space_string):
				if (last_char == ' ') and (char == ' '):
					assert space_string[char_id:char_id+1] == ' '
					del_ids.append(char_id)
				else:
					cleaned_string += char
				last_char = char
			sentence.string = cleaned_string
			for entity in sentence.entity_mentions:
				gap_both = 0
				gap_end = 0
				for del_id in del_ids:
					if del_id < entity.char_b:
						gap_both += 1
					elif entity.char_b < del_id < entity.char_e:
						gap_end += 1
				entity.char_b -= gap_both
				entity.char_e -= gap_both + gap_end
				entity.string = sentence.string[entity.char_b:entity.char_e]
			del_id_count = 0
			for id, token in enumerate(sentence.tokens):
				token.char_b -= sentence.char_b
				token.char_e -= sentence.char_b
				gap_both = 0
				gap_end = 0
				for del_id in del_ids:
					if del_id < token.char_b:
						gap_both += 1
					if token.char_b < del_id < token.char_e:
						raise IndexError
					elif (token.char_b == del_id) and (token.char_e == del_id+1):
						del_id_count += 1
						del sentence.tokens[id]
				token.char_b -= gap_both
				token.char_e -= gap_both + gap_end
				token.id -= del_id_count
				assert token.char_b is not None
				assert token.char_e is not None
			# token.text = sentence.string[token.char_b:token.char_e]


def clean_docs_chinese(doc_dicts):
	for doc_id, doc in doc_dicts.items():
		for sentence in doc.sentences:
			cleaned_string = ''
			del_ids: List[int] = []
			for char_id, char in enumerate(sentence.string):
				if char in ['\n', '\t', '\r']:
					del_ids.append(char_id)
				else:
					cleaned_string += char
			sentence.string = cleaned_string
			for entity in sentence.entity_mentions:
				gap_both = 0
				gap_end = 0
				for del_id in del_ids:
					if del_id < entity.char_b:
						gap_both += 1
					elif entity.char_b < del_id < entity.char_e:
						gap_end += 1
				entity.char_b -= gap_both
				entity.char_e -= gap_both + gap_end
				entity.string = sentence.string[entity.char_b:entity.char_e]


def parse_source_english(data_path, nlp)-> Dict[str, Document]:
	doc_dicts = {}

	# cts and un are not used	
	SgmDoc_dicts = parse_sgms_english(data_path, nlp)
	doc2entities, doc2relations, doc2events = parse_apf_docs(data_path)
	doc_dicts.update(merge_sgm_apf(SgmDoc_dicts, doc2entities, doc2relations))
	clean_docs_english(doc_dicts)
	return doc_dicts


def parse_source_chinese(data_path) -> Dict[str, Document]:
	doc_dicts = {}
	SgmDoc_dicts = parse_sgms_chinese(data_path)
	doc2entities, doc2relations, doc2events = parse_apf_docs(data_path)
	doc_dicts.update(merge_sgm_apf(SgmDoc_dicts, doc2entities, doc2relations))
	clean_docs_chinese(doc_dicts)
	return doc_dicts


if __name__ == '__main__':
	#######english########
	import argparse

	DEBUG = 1

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', default='/media/moju/data/work/resource/data/LDC2006T06/data/English/')
	parser.add_argument('--output_path', default='./output/')
	parser.add_argument('--corenlp_server', default='http://140.109.19.190')

	args = parser.parse_args()

	nlp = StanfordCoreNLP(args.corenlp_server, port=9000, lang='en')
	data_path = args.data_path
	doc_dicts = parse_source_english(data_path, nlp)

	print(len(doc_dicts))
	print(doc_dicts["AGGRESSIVEVOICEDAILY_20041101.1144"].sentences[1].char_b)
	print(doc_dicts["AGGRESSIVEVOICEDAILY_20041101.1144"].sentences[1].char_e)
	print(doc_dicts["AGGRESSIVEVOICEDAILY_20041101.1144"].sentences[1].text)

	print(str(doc_dicts["AGGRESSIVEVOICEDAILY_20041101.1144"].sentences[1].relation_mentions[0].id))
	print(str(doc_dicts["AGGRESSIVEVOICEDAILY_20041101.1144"].sentences[1].entity_mentions[0].id))
	print(str(doc_dicts["AGGRESSIVEVOICEDAILY_20041101.1144"].sentences[1].entity_mentions[0].text))
	print(str(doc_dicts["AGGRESSIVEVOICEDAILY_20041101.1144"].sentences[1].entity_mentions[0].char_b))
	print(str(doc_dicts["AGGRESSIVEVOICEDAILY_20041101.1144"].sentences[1].entity_mentions[0].char_e))

	print(doc_dicts["AGGRESSIVEVOICEDAILY_20041101.1144"].sentences[1].text[316 - 266:317 - 266])
	
	#######chinese########
	# import argparse
	#
	# parser = argparse.ArgumentParser()
	# parser.add_argument('--data_path', default='/media/moju/data/work/resource/data/LDC2006T06/data/Chinese/')
	# parser.add_argument('--output_path', default='./output/')
	# parser.add_argument('--corenlp_path', default='http://140.109.19.190')
	#
	# args = parser.parse_args()
	#
	# data_path = args.data_path
	# doc_dicts = parse_source_chinese(data_path)
	#
	# print(len(doc_dicts))
	# print(doc_dicts["CTS20001211.1300.0012"].sentences[1].text)
	# print(doc_dicts["CTS20001211.1300.0012"].sentences[1].char_b)
	# print(str(doc_dicts["CTS20001211.1300.0012"].sentences[1].relation_mentions[0].arg1.text))
	# print(str(doc_dicts["CTS20001211.1300.0012"].sentences[1].relation_mentions[0].arg1.char_b))
	# print(str(doc_dicts["CTS20001211.1300.0012"].sentences[1].relation_mentions[0].arg1.char_e))
	# print(doc_dicts["CTS20001211.1300.0012"].sentences[1].text[19:20+1])