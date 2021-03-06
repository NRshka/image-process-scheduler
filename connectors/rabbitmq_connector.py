import asyncio
from asyncio.events import AbstractEventLoop
from typing import Any, Callable, Optional
import aio_pika
import json


async def main(
    loop,
    queue_name: str,
    processing_func: Callable,
    login: str,
    passw: str,
    port: int
):
    connection: Any = await aio_pika.connect_robust(
        f"amqp://{login}:{passw}@127.0.0.1:{port}/", loop=loop,
        port=5673
    )

    async with connection:
        # Creating channel
        channel = await connection.channel()

        # Declaring queue
        queue = await channel.declare_queue(queue_name, auto_delete=True)

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    options = json.loads(message.body.decode('utf-8'))
                    await processing_func(options)


def run_async_rabbitmq_connection(
    queue_name,
    processing_function,
    login: str,
    passw: str,
    port: int,
    loop: Optional[AbstractEventLoop] = None,
):
    if not loop:
        loop = asyncio.get_event_loop()

    loop.run_until_complete(
        main(
            loop, queue_name, processing_function,
            login, passw, port
        )
    )
    # loop.close()
