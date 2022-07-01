import pandas as pd

from batch_processing_discovery.config import DEFAULT_CSV_IDS
from batch_processing_discovery.discovery import _identify_single_activity_batches


def test__identify_single_activity_batches():
    # Read input event log
    event_log = pd.read_csv("./tests/assets/event_log_1.csv")
    event_log[DEFAULT_CSV_IDS.enabled_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.enabled_time], utc=True)
    event_log[DEFAULT_CSV_IDS.start_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.start_time], utc=True)
    event_log[DEFAULT_CSV_IDS.end_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.end_time], utc=True)
    event_log['expected_id'] = event_log['expected_id'].astype('Int64')
    event_log['expected_id_gap'] = event_log['expected_id_gap'].astype('Int64')
    event_log['expected_id_size_4'] = event_log['expected_id_size_4'].astype('Int64')
    event_log['expected_id_gap_size_4'] = event_log['expected_id_gap_size_4'].astype('Int64')
    # Identify single activity batches with minimum size 2 and no gap
    _identify_single_activity_batches(event_log, DEFAULT_CSV_IDS, 2, pd.Timedelta(0))
    assert event_log[DEFAULT_CSV_IDS.batch_id].equals(event_log['expected_id'])
    # Identify single activity batches with minimum size 3 and no gap
    _identify_single_activity_batches(event_log, DEFAULT_CSV_IDS, 3, pd.Timedelta(0))
    assert event_log[DEFAULT_CSV_IDS.batch_id].equals(event_log['expected_id'])
    # Identify single activity batches with minimum size 4 and no gap
    _identify_single_activity_batches(event_log, DEFAULT_CSV_IDS, 4, pd.Timedelta(0))
    assert event_log[DEFAULT_CSV_IDS.batch_id].equals(event_log['expected_id_size_4'])
    # Identify single activity batches with minimum size 2 and no gap
    _identify_single_activity_batches(event_log, DEFAULT_CSV_IDS, 2, pd.Timedelta(5, "m"))
    assert event_log[DEFAULT_CSV_IDS.batch_id].equals(event_log['expected_id_gap'])
    # Identify single activity batches with minimum size 2 and no gap
    _identify_single_activity_batches(event_log, DEFAULT_CSV_IDS, 4, pd.Timedelta(5, "m"))
    assert event_log[DEFAULT_CSV_IDS.batch_id].equals(event_log['expected_id_gap_size_4'])
