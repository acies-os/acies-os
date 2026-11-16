from acies.controller.ns import _strip_backup_and_ctl, make_ctl_topic, parse_node_name, parse_service_name


def test_strip_backup_and_ctl():
    cases = {
        'rs10/geo/ctl': 'rs10/geo',
        'rs10/geo': 'rs10/geo',
        'geo': 'geo',
        'rs8/backup/rs10/vfm/ctl': 'rs8/backup/rs10/vfm',
        'backup/rs10/vfm/ctl': 'rs10/vfm',
    }
    for topic_in, topic_expect in cases.items():
        assert _strip_backup_and_ctl(topic_in) == topic_expect


def test_parse_node_name():
    cases = {
        'rs10/geo/ctl': 'rs10',
        'rs8/backup/rs10/vfm/ctl': 'rs8',
        'rs10/geo': 'rs10',
        'backup/rs10/vfm/ctl': 'rs10',
    }
    for topic_in, topic_expect in cases.items():
        assert parse_node_name(topic_in) == topic_expect


def test_parse_service_name():
    cases = {
        'rs10/geo/ctl': 'geo',
        'rs10/geo': 'geo',
        'rs8/backup/rs10/vfm/ctl': 'backup/rs10/vfm',
        'backup/rs10/vfm/ctl': 'vfm',
    }
    for topic_in, topic_expect in cases.items():
        assert parse_service_name(topic_in) == topic_expect


def test_make_topic():
    assert make_ctl_topic('rs10', 'geo') == 'rs10/geo/ctl'
    assert make_ctl_topic('rs10', 'vfm/ctl') == 'rs10/vfm/ctl'
    assert make_ctl_topic('rs10', 'backup/rs3/vfm/ctl') == 'rs10/backup/rs3/vfm/ctl'
