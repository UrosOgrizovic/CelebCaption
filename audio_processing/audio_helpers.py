import os
import subprocess


def create_folder(path):
    """Create folder if it doesn't exist

    Args:
        path (string): folder path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_video_duration(video_path):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)