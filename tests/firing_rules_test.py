import pandas as pd

from batch_processing_discovery.config import DEFAULT_CSV_IDS
from batch_processing_discovery.firing_rules import _get_size_distribution, _compute_features_table, _get_features, get_firing_rules


def test_get_firing_rules():
    # Read input event log
    event_log = pd.read_csv("./assets/event_log_5.csv")
    event_log[DEFAULT_CSV_IDS.enabled_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.enabled_time], utc=True)
    event_log[DEFAULT_CSV_IDS.start_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.start_time], utc=True)
    event_log[DEFAULT_CSV_IDS.end_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.end_time], utc=True)
    event_log[DEFAULT_CSV_IDS.batch_id] = event_log[DEFAULT_CSV_IDS.batch_id].astype('Int64')
    # Get the firing rules
    rules = get_firing_rules(event_log, DEFAULT_CSV_IDS)
    # Assert
    assert len(rules) == 1
    assert rules['B']['sizes'] == {1: 2, 3: 48}
    assert rules['B']['confidence'] == 1.0
    assert rules['B']['support'] == 1.0
    assert rules['B']['model'].ruleset_.rules[0].conds[0].feature == "num_queue"
    assert rules['B']['model'].ruleset_.rules[0].conds[0].val == 3


def test__compute_features_table():
    # Read input event log
    event_log = pd.read_csv("./tests/assets/event_log_4.csv")
    event_log[DEFAULT_CSV_IDS.enabled_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.enabled_time], utc=True)
    event_log[DEFAULT_CSV_IDS.start_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.start_time], utc=True)
    event_log[DEFAULT_CSV_IDS.end_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.end_time], utc=True)
    event_log[DEFAULT_CSV_IDS.batch_id] = event_log[DEFAULT_CSV_IDS.batch_id].astype('Int64')
    # Compute features table
    features_table = _compute_features_table(event_log, DEFAULT_CSV_IDS)
    # Assert features
    # 1 positive observation per batch
    assert len(features_table[features_table['outcome'] == 1]) == 4
    # 4 negative observations for the sequential with WT_ready, 2 negative for
    # the sequential with no WT_ready, and 0 for the concurrent with no WTs
    assert len(features_table[features_table['outcome'] == 0]) == 10
    assert len(features_table) == 14
    # Check that each of the positive observations are there
    positive_observations = pd.DataFrame([
        {
            DEFAULT_CSV_IDS.batch_id: 0,
            DEFAULT_CSV_IDS.batch_type: 'Sequential',
            DEFAULT_CSV_IDS.activity: "A",
            DEFAULT_CSV_IDS.resource: "Jonathan",
            'instant': pd.Timestamp("2021-01-01T09:30:00+00:00"),
            'num_queue': 3,
            't_ready': pd.Timedelta(seconds=1800),
            't_waiting': pd.Timedelta(hours=1),
            't_max_flow': pd.Timedelta(hours=1, seconds=1800),
            'day_of_week': 4,
            'day_of_month': 1,
            'hour_of_day': 9,
            'minute': 30,
            'outcome': 1
        }, {
            DEFAULT_CSV_IDS.batch_id: 1,
            DEFAULT_CSV_IDS.batch_type: 'Sequential',
            DEFAULT_CSV_IDS.activity: "C",
            DEFAULT_CSV_IDS.resource: "Jolyne",
            'instant': pd.Timestamp("2021-01-01T14:00:00+00:00"),
            'num_queue': 3,
            't_ready': pd.Timedelta(hours=1, seconds=1800),
            't_waiting': pd.Timedelta(hours=2, seconds=1800),
            't_max_flow': pd.Timedelta(hours=6),
            'day_of_week': 4,
            'day_of_month': 1,
            'hour_of_day': 14,
            'minute': 00,
            'outcome': 1
        }, {
            DEFAULT_CSV_IDS.batch_id: 2,
            DEFAULT_CSV_IDS.batch_type: 'Concurrent',
            DEFAULT_CSV_IDS.activity: "E",
            DEFAULT_CSV_IDS.resource: "Jonathan",
            'instant': pd.Timestamp("2021-01-01T16:00:00+00:00"),
            'num_queue': 3,
            't_ready': pd.Timedelta(0),
            't_waiting': pd.Timedelta(0),
            't_max_flow': pd.Timedelta(hours=8),
            'day_of_week': 4,
            'day_of_month': 1,
            'hour_of_day': 16,
            'minute': 00,
            'outcome': 1
        }, {
            DEFAULT_CSV_IDS.batch_id: 3,
            DEFAULT_CSV_IDS.batch_type: 'Sequential',
            DEFAULT_CSV_IDS.activity: "F",
            DEFAULT_CSV_IDS.resource: "Joseph",
            'instant': pd.Timestamp("2021-01-01T17:00:00+00:00"),
            'num_queue': 3,
            't_ready': pd.Timedelta(0),
            't_waiting': pd.Timedelta(seconds=1800),
            't_max_flow': pd.Timedelta(hours=9),
            'day_of_week': 4,
            'day_of_month': 1,
            'hour_of_day': 17,
            'minute': 00,
            'outcome': 1
        }
    ])
    assert features_table[features_table['outcome'] == 1].reset_index(drop=True).equals(positive_observations)
    # Check that the negative observations are there
    neg_features_batch_0 = features_table[(features_table['outcome'] == 0) & (features_table[DEFAULT_CSV_IDS.batch_id] == 0)]
    assert neg_features_batch_0['instant'].isin([
        pd.Timestamp("2021-01-01T08:30:00+00:00"),
        pd.Timestamp("2021-01-01T08:45:00+00:00"),
        pd.Timestamp("2021-01-01T09:00:00+00:00"),
        pd.Timestamp("2021-01-01T09:10:00+00:00"),
        pd.Timestamp("2021-01-01T09:20:00+00:00"),
    ]).all()
    neg_features_batch_1 = features_table[(features_table['outcome'] == 0) & (features_table[DEFAULT_CSV_IDS.batch_id] == 1)]
    assert neg_features_batch_1['instant'].isin([
        pd.Timestamp("2021-01-01T11:30:00+00:00"),
        pd.Timestamp("2021-01-01T12:00:00+00:00"),
        pd.Timestamp("2021-01-01T12:30:00+00:00"),
        pd.Timestamp("2021-01-01T13:00:00+00:00"),
        pd.Timestamp("2021-01-01T13:30:00+00:00"),
    ]).all()
    neg_features_batch_2 = features_table[(features_table['outcome'] == 0) & (features_table[DEFAULT_CSV_IDS.batch_id] == 2)]
    assert len(neg_features_batch_2) == 0
    neg_features_batch_3 = features_table[(features_table['outcome'] == 0) & (features_table[DEFAULT_CSV_IDS.batch_id] == 3)]
    assert neg_features_batch_3['instant'].isin([
        pd.Timestamp("2021-01-01T16:30:00+00:00"),
        pd.Timestamp("2021-01-01T16:45:00+00:00"),
        pd.Timestamp("2021-01-01T17:00:00+00:00")
    ]).all()


def test__get_features():
    # Read input event log
    event_log = pd.read_csv("./tests/assets/event_log_4.csv")
    event_log[DEFAULT_CSV_IDS.enabled_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.enabled_time], utc=True)
    event_log[DEFAULT_CSV_IDS.start_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.start_time], utc=True)
    event_log[DEFAULT_CSV_IDS.end_time] = pd.to_datetime(event_log[DEFAULT_CSV_IDS.end_time], utc=True)
    event_log[DEFAULT_CSV_IDS.batch_id] = event_log[DEFAULT_CSV_IDS.batch_id].astype('Int64')
    # Assert the features of the start of a batch
    features = _get_features(
        event_log=event_log,
        instant=pd.Timestamp("2021-01-01T09:30:00+00:00"),
        batch_instance=event_log[event_log[DEFAULT_CSV_IDS.batch_id] == 0],
        outcome=1,
        log_ids=DEFAULT_CSV_IDS
    )
    assert features == {
        DEFAULT_CSV_IDS.batch_id: 0,
        DEFAULT_CSV_IDS.batch_type: 'Sequential',
        DEFAULT_CSV_IDS.activity: "A",
        DEFAULT_CSV_IDS.resource: "Jonathan",
        'instant': pd.Timestamp("2021-01-01T09:30:00+00:00"),
        'num_queue': 3,
        't_ready': pd.Timedelta(seconds=1800),
        't_waiting': pd.Timedelta(hours=1),
        't_max_flow': pd.Timedelta(hours=1, seconds=1800),
        'day_of_week': 4,
        'day_of_month': 1,
        'hour_of_day': 9,
        'minute': 30,
        'outcome': 1
    }
    # Assert the features of the enabling instant in the middle of the accumulation
    features = _get_features(
        event_log=event_log,
        instant=pd.Timestamp("2021-01-01T08:45:00+00:00"),
        batch_instance=event_log[
            (event_log[DEFAULT_CSV_IDS.batch_id] == 0) &
            (event_log[DEFAULT_CSV_IDS.enabled_time] <= pd.Timestamp("2021-01-01T08:45:00+00:00"))
            ],
        outcome=0,
        log_ids=DEFAULT_CSV_IDS
    )
    assert features == {
        DEFAULT_CSV_IDS.batch_id: 0,
        DEFAULT_CSV_IDS.batch_type: 'Sequential',
        DEFAULT_CSV_IDS.activity: "A",
        DEFAULT_CSV_IDS.resource: "Jonathan",
        'instant': pd.Timestamp("2021-01-01T08:45:00+00:00"),
        'num_queue': 2,
        't_ready': pd.Timedelta(0),
        't_waiting': pd.Timedelta(seconds=900),
        't_max_flow': pd.Timedelta(seconds=2700),
        'day_of_week': 4,
        'day_of_month': 1,
        'hour_of_day': 8,
        'minute': 45,
        'outcome': 0
    }
    # Assert the features of the enabling instant in the middle of the batch ready
    features = _get_features(
        event_log=event_log,
        instant=pd.Timestamp("2021-01-01T13:00:00+00:00"),
        batch_instance=event_log[event_log[DEFAULT_CSV_IDS.batch_id] == 1],
        outcome=0,
        log_ids=DEFAULT_CSV_IDS
    )
    assert features == {
        DEFAULT_CSV_IDS.batch_id: 1,
        DEFAULT_CSV_IDS.batch_type: 'Sequential',
        DEFAULT_CSV_IDS.activity: "C",
        DEFAULT_CSV_IDS.resource: "Jolyne",
        'instant': pd.Timestamp("2021-01-01T13:00:00+00:00"),
        'num_queue': 3,
        't_ready': pd.Timedelta(seconds=1800),
        't_waiting': pd.Timedelta(hours=1, seconds=1800),
        't_max_flow': pd.Timedelta(hours=5),
        'day_of_week': 4,
        'day_of_month': 1,
        'hour_of_day': 13,
        'minute': 00,
        'outcome': 0
    }
    # Assert the features of the first enabling instant
    features = _get_features(
        event_log=event_log,
        instant=pd.Timestamp("2021-01-01T16:30:00+00:00"),
        batch_instance=event_log[
            (event_log[DEFAULT_CSV_IDS.batch_id] == 3) &
            (event_log[DEFAULT_CSV_IDS.case] == 0)
            ],
        outcome=0,
        log_ids=DEFAULT_CSV_IDS
    )
    assert features == {
        DEFAULT_CSV_IDS.batch_id: 3,
        DEFAULT_CSV_IDS.batch_type: 'Sequential',
        DEFAULT_CSV_IDS.activity: "F",
        DEFAULT_CSV_IDS.resource: "Joseph",
        'instant': pd.Timestamp("2021-01-01T16:30:00+00:00"),
        'num_queue': 1,
        't_ready': pd.Timedelta(0),
        't_waiting': pd.Timedelta(0),
        't_max_flow': pd.Timedelta(hours=8, seconds=1800),
        'day_of_week': 4,
        'day_of_month': 1,
        'hour_of_day': 16,
        'minute': 30,
        'outcome': 0
    }


def test__get_size_distribution():
    # Read input event log
    event_log = pd.read_csv("./tests/assets/event_log_3.csv")
    # Get size distribution
    size_distribution = _get_size_distribution(event_log, "A", DEFAULT_CSV_IDS, False)
    assert size_distribution == {1: 7, 3: 12, 4: 4}
    # Get size distribution taking into account the resource
    size_distribution = _get_size_distribution(event_log, ("A", "Jonathan"), DEFAULT_CSV_IDS, True)
    assert size_distribution == {1: 3, 3: 6}
    size_distribution = _get_size_distribution(event_log, ("A", "Joseph"), DEFAULT_CSV_IDS, True)
    assert size_distribution == {1: 4, 3: 6, 4: 4}
