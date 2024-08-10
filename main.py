import os
import sys
import librosa
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def list_mp3_files(directory):
    """Lists all mp3 files in the specified directory."""
    logging.info(f"Listing MP3 files in directory: {directory}")
    mp3_files = [f for f in os.listdir(directory) if f.endswith('.mp3')]
    logging.info(f"Found {len(mp3_files)} MP3 files.")
    return mp3_files

def categorize_files(files):
    """Categorizes files based on their prefixes and ensures only one file per category."""
    logging.info("Categorizing files.")
    categories = {
        'drums-other': None,
        'snare': None,
        'kick': None,
        'all-files': []
    }

    for file in files:
        categories['all-files'].append(file)
        if file.startswith('drums-other-') and categories['drums-other'] is None:
            categories['drums-other'] = file
            logging.info(f"Assigned {file} to 'drums-other' category.")
        elif file.startswith('snare-') and categories['snare'] is None:
            categories['snare'] = file
            logging.info(f"Assigned {file} to 'snare' category.")
        elif file.startswith('kick-') and categories['kick'] is None:
            categories['kick'] = file
            logging.info(f"Assigned {file} to 'kick' category.")
    
    return categories


def detect_hits(file_path, threshold=0.05):
    """Detects percussive hits in the given audio file with a lower amplitude limit."""
    logging.info(f"Detecting hits in file: {file_path}")
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Compute the RMS (Root Mean Square) energy
        rms = librosa.feature.rms(y=y)

        # Create a mask for frames where the RMS energy is above the threshold
        rms_threshold = threshold
        mask = rms > rms_threshold
        
        # Upsample the mask to match the length of `y`
        mask = np.repeat(mask, int(np.ceil(len(y) / mask.shape[1])), axis=1)
        mask = mask[:, :len(y)]  # Trim the mask to the exact length of y
        
        # Apply the mask to the signal
        y_filtered = y * mask.flatten()
        
        # Detect onsets (percussive hits)
        onset_frames = librosa.onset.onset_detect(y=y_filtered, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        logging.info(f"Detected {len(onset_times)} hits.")
        return onset_times
    except Exception as e:
        logging.error(f"Error detecting hits in file {file_path}: {e}")
        return []


def map_to_hits(directory, categories):
    """Maps categorized files to a chart-like format with detected hits."""
    logging.info("Mapping files to chart format.")
    chart = []

    for category, file in categories.items():
        if category == 'all-files':
            continue
        if file:
            file_path = os.path.join(directory, file)
            threshold = 0.05
            if category == 'kick':
                threshold = 0.1
            if category == 'drums-other':
                threshold = 0.005
            hits = detect_hits(file_path, threshold=threshold)
            prev_hit = None
            for hit in hits:
                if category == 'kick' and prev_hit and hit - prev_hit < 0.1:
                    continue
                chart.append({'instrument': category, 'time': hit})
                prev_hit = hit
                # logging.info(f"Added hit for {category} at time {hit:.2f}s.")
    
    return chart


def map_to_beat_time(hits_dict, sync_track):
    RESULUTION = 192 / 2
    ret = []
    hits_sorted_by_time = list(sorted(hits_dict, key=lambda x: x['time']))
    if not hits_sorted_by_time:
        return []

    for hit in hits_sorted_by_time[:100]:
        logger.info(f"Hit: {hit['instrument']} at {hit['time']}s")

    def find_beat_time_for_seconds(seconds_to_find):
        candidates = []
        current_8th = -RESULUTION
        current_bpm = sync_track[0] / 1000.0
        current_seconds = 0
        while True:
            current_8th += RESULUTION
            if current_8th > 0:
                current_seconds += beats_to_seconds(RESOLUTION, current_bpm)
            if current_8th in sync_track:
                # bpm change
                current_bpm = sync_track[current_8th] / 1000.0

            candidates.append({'8th': current_8th, 'seconds': current_seconds})
            if len(candidates) > 3:
                candidates.pop(0)
            if current_seconds > seconds_to_find:
                # We have passed the time
                break
    
        # Find the closest 8th from candidates
        closest = None
        for candidate in candidates:
            if closest is None or abs(candidate['seconds'] - seconds_to_find) < abs(closest['seconds'] - seconds_to_find):
                closest = candidate

        return closest['8th']

    for hit in hits_sorted_by_time:
        beat_time = find_beat_time_for_seconds(hit['time'])
        logger.info(f"Hit: {hit['instrument']} at {hit['time']}s, beat time: {beat_time}")
        ret.append({'instrument': hit['instrument'], 'time': beat_time})

    return ret


def beats_to_seconds(res_value, bpm):
    global RESOLUTION
    # resolution is 1/2th of a beat
    # if "beats" is same as resolution, it means 0.5 beats

    beats = (res_value / RESOLUTION) * 0.5
    # logger.info(f"resolution: {RESOLUTION}, res_value: {res_value}, beats: {beats}, bpm: {bpm}")

    # bps = bpm / 60
    # seconds_per_beat = 1 / bps
    # seconds = beats * seconds_per_beat

    return beats * (1 / (bpm / 60))


def seconds_to_beats(seconds, bpm):
    return seconds * (bpm / 60)


def read_sync_track(directory):
    # Find the chart file
    sync_track_file = None
    for file in os.listdir(directory):
        if file.endswith('notes.chart'):
            sync_track_file = file
            break

    if sync_track_file is None:
        logging.info("No chart file found.")
        return None
    
    logging.info(f"Reading sync track from file: {sync_track_file}")
    with open(os.path.join(directory, sync_track_file), 'r') as f:
        lines = f.readlines()

    # Find the sync track
    sync_track = []
    for line in lines:
        if sync_track and line.startswith('}'):
            break
        if line.startswith('[SyncTrack]'):
            sync_track.append(line)
        elif sync_track:
            sync_track.append(line)

    bpm_by_beat = {}
    # Trim and filter the sync track
    for line in sync_track:
        if line.startswith('[') or line.startswith('{') or line.startswith('}'):
            continue

        # Lines are like this: 0 = B 160500
        # If its like 0 = TS 4, ignore it
        parts = line.split('=')
        beat = int(parts[0].strip())
        if not parts[1].strip().startswith('B'):
            continue

        bpm = int(parts[1].split(' ')[-1].strip())
        bpm_by_beat[beat] = bpm

    logging.info(f"Found sync track: {bpm_by_beat}")
    return bpm_by_beat


def main(directory):
    # Step 1: List all mp3 files in the folder
    mp3_files = list_mp3_files(directory)

    # Step 2: Categorize files by their prefixes
    categorized_files = categorize_files(mp3_files)

    # Step 3: Read sync track
    sync_track = read_sync_track(directory)

    # Step 4: Map the files to chart format with detected hits by seconds
    hits_dict = map_to_hits(directory, categorized_files)

    hits_by_beat_time = map_to_beat_time(hits_dict, sync_track)

    return {"hits_by_beat_time": hits_by_beat_time}


def export_chart(chart, directory):
    # Find the chart file
    original_chart = None
    for file in os.listdir(directory):
        if file.endswith('notes.chart'):
            original_chart = file
            break

    hits_by_beat_time = chart['hits_by_beat_time']

    with open(os.path.join(directory, original_chart), 'r') as f:
        chart_template = f.read()

    # Add bpm values

    new_lines = """
[ExpertDrums]
{"""

    # Add bpm values

    for entry in hits_by_beat_time:
        if entry['instrument'] == 'kick':
            note = 0
        elif entry['instrument'] == 'snare':
            note = 1
        elif entry['instrument'] == 'drums-other':
            note = 2

        relative_time = int(entry['time'])

        new_lines += f"\n{relative_time} = N {note} 0"

    new_lines += "\n}"

    chart_template += new_lines

    with open(os.path.join(directory, 'output.chart'), 'w') as f:
        f.write(chart_template)


if __name__ == "__main__":
    # Replace 'your_directory_path' with the actual path to your folder
    args = sys.argv
    if len(args) < 2:
        logging.error("Please provide the directory path.")
        sys.exit(1)
    directory = args[1]
    global RESOLUTION
    RESOLUTION = 192
    logging.info("Starting the script.")
    chart = main(directory)
    export_chart(chart, directory)
