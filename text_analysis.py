import argparse
from collections import Counter
import json
import tqdm
import glob
import pandas as pd
from convokit import Speaker, Utterance, Corpus, TextParser, PolitenessStrategies

parser = argparse.ArgumentParser(
    description='Generate politeness labels using Convokit.')
parser.add_argument('-i', '--input_dir', type=str, default="dataset/train/")
parser.add_argument('-o', '--output_file', type=str, default="features.json")

REVIEWER_SPEAKER = Speaker(id="reviewer_id0", name="Reviewer0")

args = parser.parse_args()

reviews = []
text_parser = TextParser()
ps = PolitenessStrategies()

for filename in tqdm.tqdm(glob.glob(args.input_dir + "/*")):
    with open(filename, 'r') as f:
      pair_obj = json.load(f)
      reviews.append(pair_obj)

with open('features.json', 'r') as f:
  features_list = json.load(f)

features_df = pd.DataFrame.from_dict(features_list)
with open('features_dist.json', 'r') as f:
  features_list = json.load(f)

#look at specific non-politeness features
groups = [0, 118]
extracted_words = {} #most common words associated with features

sentence_types = ["arg-request_explanation", "arg-request_edit"]
# for r in groups:
#   for sentence in reviews[r]["review_sentences"]:
#     if sentence["fine"] in sentence_types:
#       print(sentence["fine"])
#       print(sentence["text"])
#   print(features_list[r]["rating"])

extracted_features = ["==HASNEGATIVE==", "==HASPOSITIVE=="]

for review in range(len(reviews)):
    print("Review " + str(review))
    for sentence in reviews[review]["review_sentences"]:
      utterance = [
        Utterance(
            id="1",
            speaker=REVIEWER_SPEAKER,
            text=sentence["text"])
      ]
      corpus = Corpus(utterances=utterance)
      corpus = text_parser.transform(corpus)
      corpus = ps.transform(corpus, markers=True)
      utt = corpus.get_utterance("1")
      for feat in extracted_features:
        feat_politeness = "feature_politeness_" + feat
        politeness_marker = "politeness_markers_" + feat
        if(utt.meta["politeness_strategies"][feat_politeness] == 1):
          # print(sentence["text"])
          # print(feat_politeness)
          # print(utt.meta["politeness_markers"][politeness_marker])
          if(feat not in extracted_words):
            extracted_words[feat] = {}
          for i in utt.meta["politeness_markers"][politeness_marker]:
            for j in i:
              if j[0] not in extracted_words[feat]:
                # extracted_words[feat][j[0]] = {}
                # extracted_words[feat][j[0]]["pol_positive"] = 0
                # extracted_words[feat][j[0]]["pol_neutral"] = 0
                # extracted_words[feat][j[0]]["pol_negative"] = 0
                # extracted_words[feat][j[0]]["none"] = 0
                # extracted_words[feat][j[0]]["total"] = 0
                extracted_words[feat][j[0]] = 0
              # extracted_words[feat][j[0]][sentence["pol"]] += 1
              # extracted_words[feat][j[0]]["total"] += 1
              extracted_words[feat][j[0]] += 1
              # if(sentence["pol"] == "none"):
              #   print(sentence["text"])
              #   print(j[0])

# print(extracted_words["==HASPOSITIVE=="])
sorted_positive = sorted(extracted_words["==HASPOSITIVE=="].items(), key = lambda x: x[1], reverse=True)
print(sorted_positive)

