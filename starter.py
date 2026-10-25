"""
starter.py
Kicks off one run of the IncidentPipeline workflow against a running
Temporal worker (see workflow.py / `temporal server start-dev`).

Usage:
    python starter.py data/sample.jpg
"""
import asyncio
import sys
import json

from temporalio.client import Client
from workflow import IncidentPipeline, PipelineInput, TASK_QUEUE


async def main(image_path: str):
    client = await Client.connect("localhost:7233")
    result = await client.execute_workflow(
        IncidentPipeline.run,
        PipelineInput(image_path=image_path),
        id=f"incident-{image_path}",
        task_queue=TASK_QUEUE,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "data/sample.jpg"
    asyncio.run(main(img))
