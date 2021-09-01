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
    default="../peer-review-discourse-dataset/data_prep/final_dataset/train/")

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


def get_features(pair_obj):
  overall_features = {}
  for fn in [get_agreeability, get_politeness]:
    overall_features.update(fn(pair_obj))
  return overall_features


def main():

  args = parser.parse_args()

  for filename in tqdm.tqdm(glob.glob(args.input_dir + "/*")):
    with open(filename, 'r') as f:
      pair_obj = json.load(f)
      features = get_features(pair_obj)


if __name__ == "__main__":
  main()
