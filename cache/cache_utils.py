import pickle
import os

def save_tracks_to_cache(tracks, cache_path):
    with open(cache_path, 'wb') as f:
        pickle.dump(tracks, f)
    print(f"Tracks saved to cache at {cache_path}")

def load_tracks_from_cache(cache_path):
    print(f"Учитавање из кеша: {cache_path}")
    with open(cache_path, 'rb') as f:
        tracks = pickle.load(f)
    print(f"Tracks loaded from cache at {cache_path}")
    return tracks

def cache_exists(cache_path):
    exists = os.path.exists(cache_path)
    print(f"Провера кеш фајла на путањи {cache_path}: {'Постоји' if exists else 'Не постоји'}")
    return exists
