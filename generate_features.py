import argparse
import collections
import glob
import json
import tqdm
from collections import Counter

from convokit import Speaker, Utterance, Corpus, TextParser, PolitenessStrategies

parser = argparse.ArgumentParser(
    description='Generate politeness labels using Convokit.')
parser.add_argument('-i', '--input_dir', type=str, default="dataset/train/")
parser.add_argument('-o', '--output_file', type=str, default="features.json")

text_parser = TextParser()
ps = PolitenessStrategies()
REVIEWER_SPEAKER = Speaker(id="reviewer_id0", name="Reviewer0")


def get_agreeability(pair_obj):
  coarse_counter = collections.Counter()
  for sentence in pair_obj["rebuttal_sentences"]:
    coarse_counter[sentence['coarse']] += 1

  if 'concur' not in coarse_counter and 'dispute' not in coarse_counter:
    return {'agreeability': None}

  return {
      "agreeability":
          coarse_counter['concur'] /
          (coarse_counter['concur'] + coarse_counter['dispute'])
  }

ARGUMENTS = "arg_evaluative arg_fact arg_other arg_request arg_social arg_structuring".split()
REQUESTS = "arg-request_clarification arg-request_edit arg-request_experiment arg-request_explanation arg-request_result arg-request_typo".split()
POLARITY = "pol_negative pol_positive pol_neutral".split()
ASPECT = "asp_motivation-impact asp_substance asp_clarity asp_meaningful-comparison asp_soundness-correctness".split()

def normalize(input_counter, key_list):
  result = {}
  total = sum(input_counter.values())
  if not total:
    result.update({"raw_"+key:0 for key in key_list})
    result.update({"normalized_"+key:0.0 for key in key_list})
  else:
    for key in key_list:
      count = input_counter[key]
      result["raw_" + key] = count
      result["normalized_" + key] = count / total

  return result


def get_review_ratios(pair_obj):
  coarse_counter = collections.Counter()
  fine_counter = collections.Counter()
  for sentence in pair_obj["review_sentences"]:
    coarse_counter[sentence["coarse"]] += 1
    if sentence["coarse"] == 'arg_request':
      fine_counter[sentence["fine"]] += 1
  normalized = normalize(coarse_counter, ARGUMENTS)
  normalized.update(normalize(fine_counter, REQUESTS))

  return normalized

def get_polarity(pair_obj):
  polarity_counter = collections.Counter()
  for sentence in pair_obj["review_sentences"]:
    polarity_counter[sentence["pol"]] += 1
  normalized = normalize(polarity_counter, POLARITY)

  return normalized

def get_aspect(pair_obj):
  aspect_counter = collections.Counter()
  for sentence in pair_obj["review_sentences"]:
    aspect_counter[sentence["asp"]] += 1
  
  normalized = normalize(aspect_counter, ASPECT)

  return normalized

def get_politeness(pair_obj):
  utterance = [
      Utterance(
          id="review_" + pair_obj["metadata"]["review_id"],
          speaker=REVIEWER_SPEAKER,
          text=" ".join(
              [sentence["text"] for sentence in pair_obj["review_sentences"]]))
  ]
  corpus = Corpus(utterances=utterance)
  corpus = text_parser.transform(corpus)
  corpus = ps.transform(corpus, markers=True)
  return corpus.get_utterances_dataframe()["meta.politeness_strategies"][0]

def get_discrete_politeness(pair_obj):
  utterances = []
  for i in range(len(pair_obj["review_sentences"])):
    utt = Utterance(
        id="review_" + pair_obj["metadata"]["review_id"] + "_" + str(i),
        speaker=REVIEWER_SPEAKER,
        text= pair_obj["review_sentences"][i]["text"]
    )
    utterances.append(utt)
  corpus = Corpus(utterances=utterances)
  corpus = text_parser.transform(corpus)
  corpus = ps.transform(corpus, markers=True)
  corpus_dict = Counter(corpus.get_utterances_dataframe().iloc[0]["meta.politeness_strategies"])
  for i in range(1, len(corpus.get_utterances_dataframe())):
    corpus_dict += Counter(corpus.get_utterances_dataframe().iloc[i]["meta.politeness_strategies"])
  for key, value in corpus.get_utterances_dataframe().iloc[0]["meta.politeness_strategies"].items():
    if key not in corpus_dict:
      corpus_dict[key] = 0
  features = {}
  for key in corpus_dict:
    features[key] = corpus_dict[key] / len(corpus.get_utterances_dataframe())
  return features

def get_metadata(pair_obj):

  keys = "review_id forum_id rating".split()
  return {
      key: pair_obj["metadata"][key] for key in keys
  }


def get_features(pair_obj):
  overall_features = {}
  # for fn in [get_metadata, get_agreeability, get_discrete_politeness, get_review_ratios, get_polarity, get_aspect]:
  for fn in [get_metadata, get_agreeability, get_discrete_politeness, get_review_ratios]:
    overall_features.update(fn(pair_obj))
  return overall_features

def main():

  args = parser.parse_args()

  feature_list = []
  for filename in tqdm.tqdm(glob.glob(args.input_dir + "/*")):
    with open(filename, 'r') as f:
      pair_obj = json.load(f)
      feature_list.append(get_features(pair_obj))

  with open(args.output_file, 'w') as f:
    json.dump(feature_list, f)


if __name__ == "__main__":
  main()
