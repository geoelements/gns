import os
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def n_files(directory: str, extension: str) -> int:
    """
    Count the number of files with a specific extension in a directory.

    Args:
        directory (str): Directory path.
        extension (str): File extension to count.

    Returns:
        int: Number of files with the specified extension.
    """
    try:
        pattern = os.path.join(directory, f"*.{extension}")
        file_count = len(glob.glob(pattern))
        return file_count
    except Exception as e:
        logger.error(
            f"Error counting files in {directory} with extension '{extension}': {e}"
        )
        return 0
