import pandas as pd

from batch_processing_discovery.config import DEFAULT_CSV_IDS
from batch_processing_discovery.discovery import _identify_single_activity_batches


def test__identify_single_activity_batches():
    # Read input event log
    event_log = pd.read_csv("./assets/event_log_1.csv")
    event_log[DEFAULT_CSV_IDS.enabled_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.enabled_time], utc=True)
    event_log[DEFAULT_CSV_IDS.start_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.start_time], utc=True)
    event_log[DEFAULT_CSV_IDS.end_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.end_time], utc=True)
    event_log['expected_batch_instance_id'] = event_log['expected_batch_instance_id'].astype('Int64')
    # Identify single activity batches
    _identify_single_activity_batches(event_log, DEFAULT_CSV_IDS, 2, pd.Timedelta(0))
    # Check if the batches were identified correctly
    assert event_log[DEFAULT_CSV_IDS.batch_id].equals(event_log['expected_batch_instance_id'])
