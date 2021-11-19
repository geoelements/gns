import json
import os


def read_metadata(data_path: str):
  """Read metadata of datasets

  Args:
    data_path: Path to metadata JSON file

  Returns:
    metadata json object
  """
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())
