import pickle
import os

def save_tracks_to_cache(tracks, cache_path):
    """
    Save tracking data to cache file.
    """
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(tracks, f)
        print(f"Tracks saved to cache at {cache_path}")
    except (OSError, IOError) as e:
        print(f"Error saving tracks to cache: {e}")

def load_tracks_from_cache(cache_path):
    """
    Load tracking data from cache file.
    """
    try:
        print(f"Loading tracks from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            tracks = pickle.load(f)
        print(f"Tracks loaded from cache at {cache_path}")
        return tracks
    except (OSError, IOError, pickle.UnpicklingError) as e:
        print(f"Error loading tracks from cache: {e}")
        return None
