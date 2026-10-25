"""
workflow.py
Temporal workflow + activities for the pipeline:

    ImageReceived -> DetectActivity -> ReportActivity -> (print/log result)

Why Temporal here: if the Gemma API call fails/rate-limits, Temporal
retries it automatically with backoff, without losing the detection
work already done. That's the "durable pipeline" story for the demo.
"""
import asyncio
from datetime import timedelta
from dataclasses import dataclass

from temporalio import workflow, activity
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.common import RetryPolicy

TASK_QUEUE = "vision-incident-queue"


@dataclass
class PipelineInput:
    image_path: str


@activity.defn
async def detect_activity(image_path: str) -> dict:
    from detector import detect_potholes
    return detect_potholes(image_path)


@activity.defn
async def report_activity(detection_result: dict) -> str:
    from gemma_report import generate_incident_report
    return generate_incident_report(detection_result)


@workflow.defn
class IncidentPipeline:
    @workflow.run
    async def run(self, input: PipelineInput) -> dict:
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=2),
            maximum_attempts=3,
        )

        detection_result = await workflow.execute_activity(
            detect_activity,
            input.image_path,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=retry_policy,
        )

        report = await workflow.execute_activity(
            report_activity,
            detection_result,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=retry_policy,
        )

        return {"detection": detection_result, "report": report}


async def run_worker():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[IncidentPipeline],
        activities=[detect_activity, report_activity],
    )
    print("[worker] listening on", TASK_QUEUE)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(run_worker())
