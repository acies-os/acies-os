import functools

import click


def common_options(f):
    """Command line options for zenoh: mode, connect, listen, topic, namespace, proc_name, deactivated."""

    @click.option(
        '--mode',
        default='peer',
        help='The zenoh session mode.',
        type=click.Choice(
            ['peer', 'client'],
            case_sensitive=False,
        ),
        show_default=True,
    )
    @click.option(
        '--connect',
        help='Endpoints to connect to.',
        type=str,
        multiple=True,
    )
    @click.option(
        '--listen',
        help='Endpoints to listen on.',
        type=str,
        multiple=True,
    )
    @click.option(
        '--topic',
        help='Topics to subscribe to.',
        type=str,
        multiple=True,
    )
    @click.option(
        '--namespace',
        help='The namespace to use.',
        type=str,
    )
    @click.option(
        '--proc_name',
        help='The process name.',
        type=str,
    )
    @click.option(
        '--deactivated',
        is_flag=True,
        default=False,
        show_default=True,
        help='Whether start the process as deactivated.',
    )
    @functools.wraps(f)
    def wrapper_common_options(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_common_options
