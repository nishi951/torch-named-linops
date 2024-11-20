#!/usr/bin/env python
#
import glob
import logging
import json


logger = logging.getLogger("torchmri")


def main():
    trace_files = glob.glob("./worker_*_trace.json")
    logger.info(f"Found trace files ", trace_files)
    merge_traces(trace_files, "merged_trace.json")


def merge_traces(input_files, output_file):
    """

    Example usage
    trace_files = glob.glob("./worker_*_trace.json")
    merge_traces(trace_files, "merged_trace.json")
    """
    all_traces = {}
    all_traces["traceEvents"] = []

    for i, trace_file in enumerate(input_files):
        with open(trace_file, "r") as f:
            trace_data = json.load(f)
            events = trace_data["traceEvents"]
            for event in events:
                event["pid"] = str(event["pid"]) + f"_{i}"

            # Important step
            all_traces["traceEvents"].extend(events)
            # Less important steps
            all_traces["schemaVersion"] = trace_data["schemaVersion"]
            all_traces["deviceProperties"] = trace_data["deviceProperties"]
    all_traces["traceName"] = str(output_file)

    # Write combined traces to a new file
    with open(output_file, "w") as f:
        json.dump(all_traces, f)


if __name__ == "__main__":
    main()
