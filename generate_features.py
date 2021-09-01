import argparse
import collections
import glob
import json
import tqdm

from convokit import Speaker, Utterance, Corpus, TextParser, PolitenessStrategies

parser = argparse.ArgumentParser(
    description='Generate politeness labels using Convokit.')
parser.add_argument(
    '-i',
    '--input_dir',
    type=str,
    default="dataset/train/")
parser.add_argument(
    '-o',
    '--output_file',
    type=str,
    default="features.json")
    
text_parser = TextParser()
ps = PolitenessStrategies()
REVIEWER_SPEAKER = Speaker(id="reviewer_id0", name="Reviewer0")


def get_agreeability(pair_obj):
  coarse_counter = collections.Counter()
  for sentence in pair_obj["rebuttal_sentences"]:
    coarse_counter[sentence['coarse']] += 1

  concur = max(coarse_counter["concur"], 1)
  dispute = max(coarse_counter["dispute"], 1)

  return {"agreeability": concur / (concur + dispute)}

def normalize(input_counter):
  result = {}
  total = sum(input_counter.values())
  for key, count in input_counter.items():
    result["raw_"+key] = count
    result["normalized_"+key] = count/total
  return result

def get_review_ratios(pair_obj):
  coarse_counter = collections.Counter()
  fine_counter = collections.Counter()
  for sentence in pair_obj["review_sentences"]:
    coarse_counter[sentence["coarse"]] += 1
    if sentence["coarse"] == 'arg_request':
      fine_counter[sentence["fine"]] += 1

  normalized = normalize(coarse_counter)
  normalized.update(normalize(fine_counter))

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

def get_metadata(pair_obj):
  return {
  "review_id":pair_obj["metadata"]["review_id"],
  "forum_id":pair_obj["metadata"]["forum_id"],
  }


def get_features(pair_obj):
  overall_features = {}
  for fn in [get_metadata, get_agreeability, get_politeness, get_review_ratios]:
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
