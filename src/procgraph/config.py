"""File locations. Paths are relative to the working directory (the repository root)."""
import os

DATA_DIR = 'data'
PROCTHOR_DATA_PATH = os.path.join(DATA_DIR, 'procthor-data', 'data.jsonl')
SENTENCE_MODEL_DIR = os.path.join(DATA_DIR, 'sentence_transformers')
PIPELINE_OUTPUT_DIR = os.path.join(DATA_DIR, 'pipeline')

SENTENCE_MODEL_NAME = 'bert-base-nli-stsb-mean-tokens'

FAST_DOWNWARD_DIR = os.environ.get(
    'FAST_DOWNWARD_DIR', os.path.join('third_party', 'fast-downward'))
