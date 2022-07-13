import pandas as pd

from batch_processing_discovery.config import EventLogIDs, BatchType


def discover_batches(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        batch_min_size: int = 2,
        max_sequential_gap: pd.Timedelta = pd.Timedelta(0)
) -> pd.DataFrame:
    """

    :param event_log:
    :param log_ids:
    :param batch_min_size:
    :param max_sequential_gap:
    :return:
    """
    batched_event_log = event_log.copy()
    # First phase: identify single activity batches
    _identify_single_activity_batches(batched_event_log, log_ids, batch_min_size, max_sequential_gap)
    # Second phase: identify subprocess batches
    # _identify_subprocess_batches(batched_event_log, log_ids, max_sequential_gap)
    # Third phase: classify batch type and assign an ID
    _classify_batch_types(batched_event_log, log_ids)
    # Return event log with batch information
    return batched_event_log


def _identify_single_activity_batches(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        batch_min_size: int,
        max_sequential_gap: pd.Timedelta
):
    batches = []
    # Group all the activity instances of the same activity and resource
    for _, events in event_log.groupby([log_ids.resource, log_ids.activity]):
        # Sweep line algorithm
        batch_instance = []
        for index, event in events.sort_values([log_ids.start_time]).iterrows():
            if len(batch_instance) == 0:
                # Add first event to the batch
                batch_instance = [index]
                batch_instance_start = event[log_ids.start_time]
                batch_instance_end = event[log_ids.end_time]
            else:
                # Batch detection in process
                if (event[log_ids.enabled_time] <= batch_instance_start and
                        (event[log_ids.start_time] - batch_instance_end) <= max_sequential_gap):
                    # Add event to batch
                    batch_instance += [index]
                    # Update batch end if necessary
                    batch_instance_end = max(batch_instance_end, event[log_ids.end_time])
                else:
                    # Event not in batch: create current one if it fulfill the constraints
                    if len(batch_instance) >= batch_min_size:
                        batches += [batch_instance]
                    # Start another batch instance candidate with new event
                    batch_instance = [index]
                    batch_instance_start = event[log_ids.start_time]
                    batch_instance_end = event[log_ids.end_time]
        # Process last iteration
        if len(batch_instance) >= batch_min_size:
            batches += [batch_instance]
    # Assign a batch ID for each group
    event_log[log_ids.batch_id] = pd.NA
    indexes, ids = [], []
    for batch_id, batch_indexes in enumerate(batches):
        # Flatten list of batch indexes (lists)
        indexes += batch_indexes
        # For each index of this batch, add the batch ID
        ids += [batch_id] * len(batch_indexes)
    # Set IDs for batched columns
    event_log.loc[indexes, log_ids.batch_id] = ids
    event_log[log_ids.batch_id] = event_log[log_ids.batch_id].astype('Int64')


def _classify_batch_types(event_log: pd.DataFrame, log_ids: EventLogIDs):
    # Set batch type to NA
    event_log[log_ids.batch_type] = pd.NA
    # Check the type of each batch and save the indexes of its events
    indexes, types = [], []
    for batch_id, batch_events in event_log[~pd.isna(event_log[log_ids.batch_id])].groupby([log_ids.batch_id]):
        indexes += list(batch_events.index)
        if _is_parallel_batch(batch_events, log_ids):
            types += [BatchType.parallel] * len(batch_events)
        elif _is_concurrent_batch(batch_events, log_ids):
            types += [BatchType.concurrent] * len(batch_events)
        else:
            types += [BatchType.sequential] * len(batch_events)
    # Set the batch types
    event_log.loc[indexes, log_ids.batch_type] = types


def _is_parallel_batch(batch_events: pd.DataFrame, log_ids: EventLogIDs) -> bool:
    # If all events share start and end time, is parallel
    return (len(batch_events[log_ids.start_time].unique()) == 1 and
            len(batch_events[log_ids.end_time].unique()) == 1)


def _is_concurrent_batch(batch_events: pd.DataFrame, log_ids: EventLogIDs) -> bool:
    concurrent = False
    # Sort events and take the start times
    sorted_batch_events = batch_events.sort_values([log_ids.start_time, log_ids.end_time])
    starts = list(sorted_batch_events[log_ids.start_time])
    # Go over the end times checking if they do not overlap with the next batched event
    for i, end in enumerate(sorted_batch_events[log_ids.end_time]):
        if len(starts) > (i + 1) and starts[i + 1] < end:
            # Overlapping, thus concurrent
            concurrent = True
    # Return result
    return concurrent
