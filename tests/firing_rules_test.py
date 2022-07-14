import pandas as pd

from batch_processing_discovery.config import DEFAULT_CSV_IDS
from batch_processing_discovery.firing_rules import _get_size_distribution


def test__get_size_distribution():
    # Read input event log
    event_log = pd.read_csv("./assets/event_log_3.csv")
    # Get size distribution
    size_distribution = _get_size_distribution(event_log, "A", DEFAULT_CSV_IDS, False)
    assert size_distribution == {1: 7, 3: 12, 4: 4}
    # Get size distribution taking into account the resource
    size_distribution = _get_size_distribution(event_log, ("A", "Jonathan"), DEFAULT_CSV_IDS, True)
    assert size_distribution == {1: 3, 3: 6}
    size_distribution = _get_size_distribution(event_log, ("A", "Joseph"), DEFAULT_CSV_IDS, True)
    assert size_distribution == {1: 4, 3: 6, 4: 4}
