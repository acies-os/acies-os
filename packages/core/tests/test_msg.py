import json

from acies.core.msg import AciesMsg


def test_types(snapshot):
    msgs = []
    for t in ['i16', 'i32', 'i64', 'f64']:
        m = AciesMsg.new_array_msg([1, 2, 3, 4], 'rs10/geo/ctl', {'key': 'val'}, data_type=t, timestamp_ns=1)
        msgs.append(m.to_dict())

    m = AciesMsg.new_heartbeat('rs10/geo/ctl', {'key': 'val'}, timestamp_ns=1)
    msgs.append(m.to_dict())

    for t in ['set', 'get', 'topic', 'reply']:
        m = AciesMsg.new_ctl_msg(t, 'rs10/geo/ctl', {'key': 'val'}, {'key': 'val'}, timestamp_ns=1)
        msgs.append(m.to_dict())

    m = AciesMsg.new_json_msg('rs10/geo/ctl', {'key': 'val'}, {'key': 'val'}, timestamp_ns=1)
    msgs.append(m.to_dict())

    expected = snapshot()
    msgs = json.dumps(msgs, indent=4)
    assert expected == msgs
