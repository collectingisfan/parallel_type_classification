from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'
MODEL_DIR = ROOT_DIR / 'checkpoint'

LOG_DIR = ROOT_DIR / 'logs'
# LOG_DIR = Path('/root/tf-logs')

RAW_TRAIN_DATA = 'train.txt'
RAW_VALID_DATA = 'valid.txt'
RAW_TEST_DATA = 'test.txt'
BERT_MODEL_NAME = 'google-bert/bert-base-uncased'

LABELS_FILE = 'labels.txt'

BATCH_SIZE = 16

LEARNING_RATE = 1e-5
EPOCHS = 10

SAVE_STEPS = 50
