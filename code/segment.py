
#download here: https://nlp.stanford.edu/software/segmenter.html#Download
#docs: http://www.nltk.org/_modules/nltk/tokenize/stanford_segmenter.html

from nltk.tokenize.stanford_segmenter import StanfordSegmenter
import os

os.chdir("/Users/samski/Downloads/un_subset/chinese")


segmenter = StanfordSegmenter(path_to_jar="/Users/samski/Downloads/stanford-segmenter-2017-06-09/stanford-segmenter-3.8.0.jar", path_to_sihan_corpora_dict="/Users/samski/Downloads/stanford-segmenter-2017-06-09/data", java_class='edu.stanford.nlp.ie.crf.CRFClassifier', path_to_model="/Users/samski/Downloads/stanford-segmenter-2017-06-09/data/pku.gz", path_to_dict="/Users/samski/Downloads/stanford-segmenter-2017-06-09/data/dict-chris6.ser.gz", path_to_slf4j="/Users/samski/Downloads/stanford-segmenter-2017-06-09/stanford-segmenter-3.8.0.jar")

# walk through the chinese directory
for count, filename in enumerate(os.listdir("/Users/samski/Downloads/un_subset/chinese")):
	# print("/Users/samski/Downloads/un_subset/chinese/" + filename)

	outfile = open(str(count)+"segmented_chinese.txt", mode='w')

	result = segmenter.segment_file(filename)
	# outfile.write(result.encode('UTF-8'))
	outfile.write(result)
	outfile.close()