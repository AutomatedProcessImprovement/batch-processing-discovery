import pandas as pd

from batch_processing_discovery.config import DEFAULT_CSV_IDS
from batch_processing_discovery.firing_rules import _get_size_distribution, get_firing_rules, _get_duration_distribution


def test_get_firing_rules():
    # Read input event log
    event_log = pd.read_csv("./tests/assets/event_log_5.csv")
    event_log[DEFAULT_CSV_IDS.enabled_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.enabled_time], utc=True)
    event_log[DEFAULT_CSV_IDS.start_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.start_time], utc=True)
    event_log[DEFAULT_CSV_IDS.end_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.end_time], utc=True)
    event_log[DEFAULT_CSV_IDS.batch_id] = event_log[DEFAULT_CSV_IDS.batch_id].astype('Int64')
    # Get the firing rules
    rules = get_firing_rules(event_log, DEFAULT_CSV_IDS)
    # Assert
    assert len(rules) == 1
    rule = rules[0]
    assert rule['activity'] == 'B'
    assert rule['resources'] == ['Jolyne']
    assert rule['type'] == 'Sequential'
    assert rule['batch_frequency'] == 48 / 50
    assert rule['size_distribution'] == {1: 2, 3: 48}
    assert rule['duration_distribution'] == {3: 0.5}


def test__get_size_distribution():
    # Read input event log
    event_log = pd.read_csv("./tests/assets/event_log_3.csv")
    # Get size distribution
    filtered_event_log = event_log[event_log[DEFAULT_CSV_IDS.activity] == "A"]
    size_distribution = _get_size_distribution(filtered_event_log, DEFAULT_CSV_IDS)
    assert size_distribution == {1: 7, 3: 12, 4: 4}
    # Get size distribution taking into account the resource
    filtered_event_log = event_log[(event_log[DEFAULT_CSV_IDS.activity] == "A") & (event_log[DEFAULT_CSV_IDS.resource] == "Jonathan")]
    size_distribution = _get_size_distribution(filtered_event_log, DEFAULT_CSV_IDS)
    assert size_distribution == {1: 3, 3: 6}
    filtered_event_log = event_log[(event_log[DEFAULT_CSV_IDS.activity] == "A") & (event_log[DEFAULT_CSV_IDS.resource] == "Joseph")]
    size_distribution = _get_size_distribution(filtered_event_log, DEFAULT_CSV_IDS)
    assert size_distribution == {1: 4, 3: 6, 4: 4}


def test__get_duration_distribution():
    # Read input event log
    event_log = pd.read_csv("./tests/assets/event_log_6.csv")
    event_log[DEFAULT_CSV_IDS.enabled_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.enabled_time], utc=True)
    event_log[DEFAULT_CSV_IDS.start_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.start_time], utc=True)
    event_log[DEFAULT_CSV_IDS.end_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.end_time], utc=True)
    event_log[DEFAULT_CSV_IDS.batch_id] = event_log[DEFAULT_CSV_IDS.batch_id].astype('Int64')
    # Get size distribution
    filtered_event_log = event_log[event_log[DEFAULT_CSV_IDS.activity] == "B"]
    duration_distribution = _get_duration_distribution(filtered_event_log, DEFAULT_CSV_IDS)
    assert duration_distribution == {3: 0.6666666666666666}
    # Get size distribution taking into account the resource
    filtered_event_log = event_log[(event_log[DEFAULT_CSV_IDS.activity] == "B") & (event_log[DEFAULT_CSV_IDS.resource] == "Jolyne")]
    duration_distribution = _get_duration_distribution(filtered_event_log, DEFAULT_CSV_IDS)
    assert duration_distribution == {3: 0.5625}
    filtered_event_log = event_log[(event_log[DEFAULT_CSV_IDS.activity] == "B") & (event_log[DEFAULT_CSV_IDS.resource] == "Jotaro")]
    duration_distribution = _get_duration_distribution(filtered_event_log, DEFAULT_CSV_IDS)
    assert duration_distribution == {3: 0.75}
