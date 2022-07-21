import random
from typing import Union

import numpy as np
import pandas as pd

from .config import EventLogIDs, BatchType
from .features_table import _compute_features_table
from .rules import _get_rules


def get_firing_rules(event_log: pd.DataFrame, log_ids: EventLogIDs, resource_aware: bool = False) -> dict:
    """
    Discover the rules that better describe the firing of the batches in [event_log].

    :param event_log:       event log with the batch information already discovered.
    :param log_ids:         mapping with the IDs of each column in the dataset.
    :param resource_aware:  if True, take into the account both the resource and the executed activity
                            for the rules discovery.
    :return: a dictionary with the batched activity (and the resource if [resource_aware] is set to True)
    as key, and a dict with the frequency of each size of the batch, the rules, their confidence, and their
    support as value.
    """
    # Parse features table to transform its values
    parsed_features_table = _compute_features_table(event_log, log_ids)
    parsed_features_table['instant'] = parsed_features_table['instant'].astype(np.int64) / 10 ** 9
    parsed_features_table['t_ready'] = parsed_features_table['t_ready'].apply(lambda t: t.total_seconds())
    parsed_features_table['t_waiting'] = parsed_features_table['t_waiting'].apply(lambda t: t.total_seconds())
    parsed_features_table['t_max_flow'] = parsed_features_table['t_max_flow'].apply(lambda t: t.total_seconds())
    # Prepare datasets based on the type
    if resource_aware:
        to_drop = [log_ids.batch_id, log_ids.batch_type]
        keys = [log_ids.activity, log_ids.resource]
    else:
        to_drop = [log_ids.batch_id, log_ids.batch_type, log_ids.resource]
        keys = [log_ids.activity]
    batch_groups = parsed_features_table.drop(to_drop, axis=1).groupby(keys)
    # Calculate activation rules per batch group
    rules = {}
    for (group_key, batch_group) in batch_groups:
        size_distribution = _get_size_distribution(event_log, group_key, log_ids, resource_aware)
        filtered_group = batch_group.drop(keys, axis=1)
        if len(filtered_group['outcome'].unique()) > 1:
            discovered_rules = _get_rules(filtered_group, 'outcome')
            if len(discovered_rules) > 0:
                discovered_rules['sizes'] = size_distribution
                rules[group_key] = discovered_rules
        if group_key not in rules:
            rules[group_key] = {'sizes': size_distribution}
    return rules


def _get_size_distribution(event_log: pd.DataFrame, key: Union[str, tuple], log_ids: EventLogIDs, resource_aware: bool) -> dict:
    """
    Get the distribution of sizes of executions of the batch identified by the keys [key].

    :param event_log:       event log with all the activity instances.
    :param key:             the activity (or the activity and resource if [resource_aware] is True)
                            identifying the batch.
    :param log_ids:         mapping with the IDs of each column in the dataset.
    :param resource_aware:  if True, take into the account both the resource and the executed activity.

    :return: a dict with the features of this batch instance.
    """
    sizes = {}
    # Get activity instances of the activity of this batch (and performed by the resource if necessary)
    if resource_aware:
        (activity, resource) = key
        filtered_event_log = event_log[(event_log[log_ids.activity] == activity) & (event_log[log_ids.resource] == resource)]
    else:
        filtered_event_log = event_log[event_log[log_ids.activity] == key]
    # For each batch, increase one the count of their size
    batched_executions = filtered_event_log[~pd.isna(filtered_event_log[log_ids.batch_id])]
    for batch_id, events in batched_executions.groupby([log_ids.batch_id]):
        batch_size = len(events)
        sizes[batch_size] = sizes.get(batch_size, 0) + len(events)
    # Add count of single executions
    sizes[1] = len(filtered_event_log) - len(batched_executions)
    # Return size distribution
    return sizes
