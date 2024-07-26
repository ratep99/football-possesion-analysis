import pickle

def load_detections_from_cache(cache_path):
    """Loads detections from a cache file if it exists."""
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def save_detections_to_cache(detections, cache_path):
    """Saves tracks to a cache file."""
    with open(cache_path, 'wb') as f:
        pickle.dump(detections, f)

def load_tracks_from_cache(cache_path):
    """Loads detections from a cache file if it exists."""
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def save_tracks_to_cache(tracks, cache_path):
    """Saves tracks to a cache file."""
    with open(cache_path, 'wb') as f:
        pickle.dump(tracks, f)
