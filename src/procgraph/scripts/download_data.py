"""Download the ProcTHOR-10k houses and the sentence-BERT model into data/.

    uv run python -m procgraph.scripts.download_data
"""
import json
import os

from procgraph import config


def download_procthor_houses(path=config.PROCTHOR_DATA_PATH):
    """Save the ProcTHOR-10k training houses as JSON lines, one house per line."""
    import prior

    os.makedirs(os.path.dirname(path), exist_ok=True)
    houses = prior.load_dataset('procthor-10k')['train']
    with open(path, 'w') as f:
        for house in houses:
            f.write(json.dumps(house) + '\n')


def download_sentence_model(model_dir=config.SENTENCE_MODEL_DIR):
    from sentence_transformers import SentenceTransformer

    os.makedirs(model_dir, exist_ok=True)
    SentenceTransformer(config.SENTENCE_MODEL_NAME).save(model_dir)


if __name__ == '__main__':
    download_procthor_houses()
    download_sentence_model()
