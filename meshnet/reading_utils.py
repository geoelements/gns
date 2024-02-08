import os
import json

def read_metadata(data_path: str,
                  purpose: str,
                  file_name: str = "metadata.json"):
  """Read metadata of datasets

  Args:
    data_path (str): Path to metadata JSON file
    purpose (str): Optional str whether "train" or "rollout"
    file_name (str): Name of metadata file

  Returns:
    metadata json object
  """
  try:
    with open(os.path.join(data_path, file_name), 'rt') as fp:
      # New version use separate metadata for `train` and `rollout`.
      metadata = json.loads(fp.read())[purpose]

  except:
    with open(os.path.join(data_path, file_name), 'rt') as fp:
      # The previous format of the metadata does not distinguish the purpose of metadata
      metadata = json.loads(fp.read())

  return metadata
