import random
from typing import Union

import numpy as np
import pandas as pd

from .config import EventLogIDs
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


def _compute_features_table(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        num_batch_ready_negative_events: int = 2,
        num_batch_enabled_negative_events: int = 2
) -> pd.DataFrame:
    """
    Create a DataFrame with the features of the batch-related events, classifying them into events that activate the batch and events
    that does not activate the batch.

    :param event_log:                           event log with the batch information already discovered.
    :param log_ids:                             mapping with the IDs of each column in the dataset.
    :param num_batch_ready_negative_events:     number of non-firing instants in between the batch enablement and firing.
    :param num_batch_enabled_negative_events:   number of non-firing instants from the enablement times of each case in the batch.
    :return: A Dataframe with the features of the events activating a batch.
    """
    # Event log with events related to batches
    batch_log = event_log[~pd.isna(event_log[log_ids.batch_id])]
    # Register firing feature for each single activity that is not executed as a batch?
    # - I was thinking of this option, but kinda discarded it. It could be good to also take as observations the firing of the
    #   batched activity as a single activity (not batched) if it is executed some times individually. In this way we could
    #   discover rules like "try to batch it but meanwhile the WT is not more than 1 day", so single activities could be fired
    #   not as part of a batch because the WT is 1 day.
    # - However, in order to be fruitful we have to assume that all the executions (individual and batched) are planned to be
    #   part of a batch, but the individual ones got fired because the firing rule was activated before accumulating more than
    #   1 instance.
    # - The single instances could be executed individually just because the resource wanted, and not related to the firing rule
    #   being activated. Thus, consider them without knowing if they were thought to be a batch could hinder the rules discovery.
    # Register features for each batch instance
    features = []
    for (key, batch_instance) in batch_log.groupby([log_ids.batch_id]):
        batch_instance_start = batch_instance[log_ids.start_time].min()
        # Get features of the instant activating the batch instance
        features += [
            _get_features(
                event_log,
                batch_instance_start,
                batch_instance,
                1,  # Batch fired at this instant
                log_ids
            )
        ]
        # Get features of non-activating instants
        non_activating_instants = []
        # 1 - X events in between the ready time of the batch
        batch_instance_enabled = batch_instance[log_ids.enabled_time].max()
        non_activating_instants += pd.date_range(
            start=batch_instance_enabled,
            end=batch_instance_start,
            periods=num_batch_ready_negative_events + 2
        )[1:-1].tolist()
        # 2 - Instants per enablement time of each case
        enable_times = [instant for instant in batch_instance[log_ids.enabled_time] if instant < batch_instance_start]
        non_activating_instants += random.sample(enable_times, min(len(enable_times), num_batch_enabled_negative_events))
        # 3 - Obtain the features per instant
        for instant in non_activating_instants:
            if instant < batch_instance_start:
                # Discard the batch cases enabled after the current instant, and then calculate the features of the remaining cases.
                features += [
                    _get_features(
                        event_log,
                        instant,
                        batch_instance[batch_instance[log_ids.enabled_time] <= instant],
                        0,
                        log_ids
                    )
                ]
    return pd.DataFrame(data=features)


def _get_features(
        event_log: pd.DataFrame,
        instant: pd.Timestamp,
        batch_instance: pd.DataFrame,
        outcome: int,
        log_ids: EventLogIDs
) -> dict:
    """
    Get the features to discover activation rules of a specific instant [instant] in a batch instance [batch_instance].

    :param event_log:       event log with all the activity instances.
    :param instant:         instant of the event to register.
    :param batch_instance:  DataFrame with the activity instances of the batch instance.
    :param outcome:         integer indicating the outcome of this instant, 1 if the batch is fired, 0 if not.
    :param log_ids:         mapping with the IDs of each column in the dataset.

    :return: a dict with the features of this batch instance.
    """
    batch_id = batch_instance[log_ids.batch_id].iloc[0]
    batch_type = batch_instance[log_ids.batch_type].iloc[0]
    activity = batch_instance[log_ids.activity].iloc[0]
    resource = batch_instance[log_ids.resource].iloc[0]
    num_queue = len(batch_instance)
    t_ready = instant - batch_instance[log_ids.enabled_time].max()
    t_waiting = instant - batch_instance[log_ids.enabled_time].min()
    case_ids = batch_instance[log_ids.case].unique()
    t_max_flow = (instant - event_log[event_log[log_ids.case].isin(case_ids)][log_ids.start_time].min())
    day_of_week = instant.day_of_week
    day_of_month = instant.day
    hour_of_day = instant.hour
    minute_of_day = instant.minute
    # Return the features dict
    return {
        log_ids.batch_id: batch_id,
        log_ids.batch_type: batch_type,
        log_ids.activity: activity,
        log_ids.resource: resource,
        'instant': instant,
        'num_queue': num_queue,
        't_ready': t_ready,
        't_waiting': t_waiting,
        't_max_flow': t_max_flow,
        'day_of_week': day_of_week,
        'day_of_month': day_of_month,
        'hour_of_day': hour_of_day,
        'minute': minute_of_day,
        'outcome': outcome
    }


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
