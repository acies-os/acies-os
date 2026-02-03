import logging
import queue
import time

import click
from acies.core import Service, common_options, get_zconf, init_logger

logger = logging.getLogger('acies.player')


class Echo(Service):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle(self):
        try:
            topic, msg = self.msg_q.get_nowait()
        except queue.Empty:
            return

        logger.info('--------------------------------------------------')
        logger.info(f'Received message at topic {topic}: {msg.to_dict()}')
        x = msg.to_json()
        logger.debug(f'JSON: {x}')
        logger.debug(f'JSON length: {len(x)}')
        logger.debug(f'JSON str bytes: {x.encode()}')
        logger.debug(f'JSON str bytes length: {len(x.encode())}')
        logger.debug(f'Postcard length: {len(msg.to_bytes())}')

    def run(self):
        while True:
            self.handle()
            time.sleep(0.1)


@click.command()
@common_options
def main(
    mode,
    connect,
    listen,
    topic,
    namespace,
    proc_name,
    deactivated,
):
    conf = get_zconf(mode, connect, listen)
    init_logger('echo.log')

    echo = Echo(
        conf=conf,
        namespace=namespace,
        proc_name=proc_name,
        topic=topic,
        deactivated=deactivated,
    )
    echo.start()


if __name__ == '__main__':
    main()
