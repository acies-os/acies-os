def _strip_backup_and_ctl(topic: str) -> str:
    """Remove prefix `backup/` and suffix `/ctl`"""
    parts = topic.split('/')

    if parts and parts[0] == 'backup':
        parts = parts[1:]

    if parts and parts[-1] == 'ctl':
        parts = parts[:-1]

    return '/'.join(parts)


def parse_node_name(topic: str) -> str:
    """Get the node name portion of a topic"""
    node_name = _strip_backup_and_ctl(topic)
    node_name = node_name.split('/')[0]
    return node_name


def parse_service_name(topic: str) -> str:
    """Get the service name portion of a topic"""
    service_name = _strip_backup_and_ctl(topic)
    if '/' in service_name:
        parts = service_name.split('/')
        service_name = '/'.join(parts[1:])
    return service_name


def make_ctl_topic(node: str, service: str) -> str:
    """Make sure a topic ends with `/ctl`"""
    if not service.endswith('/ctl'):
        service = service + '/ctl'
    return f'{node}/{service}'
