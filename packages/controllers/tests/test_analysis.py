from acies.controller.analysis import ServiceState, failover_check_node, group_states_by_node, noise_check_node


def test_group_states_by_node():
    states = [
        ServiceState(1, 'heartbeat', 'rs10', 'geo', 1234567890, {'deactivated': False}),
        ServiceState(2, 'heartbeat', 'rs10', 'mic', 1234567890, {'deactivated': False}),
        ServiceState(3, 'heartbeat', 'rs10', 'vfm', 1234567890, {'deactivated': False}),
        ServiceState(4, 'heartbeat', 'rs10', 'backup/rs3/vfm', 1234567890, {'deactivated': False}),
        ServiceState(5, 'heartbeat', 'rs3', 'geo', 1234567890, {'deactivated': False}),
        ServiceState(6, 'heartbeat', 'rs3', 'vfm-geo', 1234567890, {'deactivated': False}),
        ServiceState(7, 'heartbeat', 'rs3', 'backup/rs10/vfm-mic', 1234567890, {'deactivated': False}),
    ]
    groups = group_states_by_node(states)
    expect = {
        'rs10': sorted(
            [
                ServiceState(1, 'heartbeat', 'rs10', 'geo', 1234567890, {'deactivated': False}),
                ServiceState(2, 'heartbeat', 'rs10', 'mic', 1234567890, {'deactivated': False}),
                ServiceState(3, 'heartbeat', 'rs10', 'vfm', 1234567890, {'deactivated': False}),
                ServiceState(7, 'heartbeat', 'rs3', 'backup/rs10/vfm-mic', 1234567890, {'deactivated': False}),
            ],
            key=lambda x: x.id,
        ),
        'rs3': sorted(
            [
                ServiceState(5, 'heartbeat', 'rs3', 'geo', 1234567890, {'deactivated': False}),
                ServiceState(6, 'heartbeat', 'rs3', 'vfm-geo', 1234567890, {'deactivated': False}),
                ServiceState(4, 'heartbeat', 'rs10', 'backup/rs3/vfm', 1234567890, {'deactivated': False}),
            ],
            key=lambda x: x.id,
        ),
    }
    assert groups == expect


def test_failover_check_node():
    deps = {
        'vfm': {'geo', 'mic'},
        'vfm-geo': {'geo'},
        'vfm-mic': {'mic'},
    }
    ctl_topic = 'cp/controller/ctl'

    all_healthy = [
        # live services
        ServiceState(1, 'heartbeat', 'rs10', 'geo', 1, {'deactivated': False}),
        ServiceState(2, 'heartbeat', 'rs10', 'mic', 1, {'deactivated': False}),
        ServiceState(3, 'heartbeat', 'rs10', 'vfm', 1, {'deactivated': False}),
    ]
    assert failover_check_node(ctl_topic, 'rs10', all_healthy, deps) == []

    healthy_with_backups = [
        # live services
        ServiceState(1, 'heartbeat', 'rs10', 'geo', 1, {'deactivated': False}),
        ServiceState(2, 'heartbeat', 'rs10', 'mic', 1, {'deactivated': False}),
        ServiceState(3, 'heartbeat', 'rs10', 'vfm', 1, {'deactivated': False}),
        # backup services
        ServiceState(4, 'heartbeat', 'rs10', 'rs10/vfm-geo', 1, {'deactivated': True}),
        ServiceState(5, 'heartbeat', 'rs3', 'backup/rs10/vfm', 1, {'deactivated': True}),
    ]
    assert failover_check_node(ctl_topic, 'rs10', healthy_with_backups, deps) == []

    vfm_to_vfm_geo = [
        # live services
        ServiceState(1, 'heartbeat', 'rs10', 'geo', 1, {'deactivated': False}),
        ServiceState(3, 'heartbeat', 'rs10', 'vfm', 1, {'deactivated': False}),
        # backup services
        ServiceState(4, 'heartbeat', 'rs10', 'vfm-geo', 1, {'deactivated': True}),
        ServiceState(5, 'heartbeat', 'rs3', 'backup/rs10/vfm', 1, {'deactivated': True}),
    ]
    expect = [
        # deactivate vfm because of missing mic
        {'topic': 'rs10/vfm/ctl', 'payload': {'deactivated': True}},
        # activate vfm-geo
        {'topic': 'rs10/vfm-geo/ctl', 'payload': {'deactivated': False}},
    ]
    assert failover_check_node(ctl_topic, 'rs10', vfm_to_vfm_geo, deps) == expect

    vfm_to_vfm_on_another_node = [
        # live services
        ServiceState(1, 'heartbeat', 'rs10', 'geo', 1, {'deactivated': False}),
        ServiceState(3, 'heartbeat', 'rs10', 'mic', 1, {'deactivated': False}),
        # backup services
        ServiceState(4, 'heartbeat', 'rs10', 'vfm-geo', 1, {'deactivated': True}),
        ServiceState(5, 'heartbeat', 'rs3', 'backup/rs10/vfm', 1, {'deactivated': True}),
    ]
    expect = [
        {'topic': 'rs3/backup/rs10/vfm/ctl', 'payload': {'deactivated': False}},
    ]
    assert failover_check_node(ctl_topic, 'rs10', vfm_to_vfm_on_another_node, deps) == expect

    case1 = [
        ServiceState(id=3782, kind='heartbeat', node='rs10', service='geo', timestamp_ns=1724553089508893180, state={}),
        ServiceState(
            id=3768,
            kind='heartbeat',
            node='rs10',
            service='vfm_geo',
            timestamp_ns=1724553085726314292,
            state={'deactivated': True},
        ),
        ServiceState(
            id=3778,
            kind='heartbeat',
            node='rs10',
            service='vfm',
            timestamp_ns=1724553088841453903,
            state={'deactivated': False},
        ),
    ]

    assert failover_check_node(ctl_topic, 'rs10', case1, deps) == []


# def test_noise_check_node():
#     # Dict: noise energy threshold for each node - mod
#     threshold_deps = {f'rs{id}': {'geo': 500, 'mic': 500} for id in range(1, 11)}  # 500 for testing
#     deps = {
#         **threshold_deps,
#         **{
#             'vfm': {'geo', 'mic'},
#             'vfm-geo': {'geo'},
#             'vfm-mic': {'mic'},
#         },
#     }
#
#     ctl_topic = 'cp/controller/ctl'
#
#     test_cases = {
#         'single_clean': {
#             'input': [
#                 # live services
#                 ServiceState(1, 'heartbeat', 'rs1', 'geo', 1, {'deactivated': False}),
#                 ServiceState(2, 'heartbeat', 'rs1', 'mic', 1, {'deactivated': False}),
#                 ServiceState(3, 'heartbeat', 'rs1', 'vfm', 1, {'deactivated': False}),
#                 ServiceState(4, 'heartbeat', 'rs1', 'vfm-geo', 1, {'deactivated': True}),
#                 ServiceState(5, 'heartbeat', 'rs1', 'vfm-mic', 1, {'deactivated': True}),
#                 ServiceState(6, 'noise_detector', 'rs1', 'nd', 1, {'mic': 100, 'geo': 100}),
#             ],
#             'expected': [],
#         },
#         'single_mic_noisy': {
#             'input': [
#                 # live services
#                 ServiceState(1, 'heartbeat', 'rs1', 'geo', 1, {'deactivated': False}),
#                 ServiceState(2, 'heartbeat', 'rs1', 'mic', 1, {'deactivated': False}),
#                 ServiceState(3, 'heartbeat', 'rs1', 'vfm', 1, {'deactivated': False}),
#                 ServiceState(4, 'heartbeat', 'rs1', 'vfm-geo', 1, {'deactivated': True}),
#                 ServiceState(5, 'heartbeat', 'rs1', 'vfm-mic', 1, {'deactivated': True}),
#                 ServiceState(6, 'noise_detector', 'rs1', 'nd', 1, {'mic': 1000, 'geo': 100}),
#             ],
#             'expected': [
#                 {'topic': 'rs1/vfm/ctl', 'payload': {'deactivated': True}},
#                 {'topic': 'rs1/vfm-geo/ctl', 'payload': {'deactivated': False}},
#             ],
#         },
#         'single_geo_noisy': {
#             'input': [
#                 # live services
#                 ServiceState(1, 'heartbeat', 'rs1', 'geo', 1, {'deactivated': False}),
#                 ServiceState(2, 'heartbeat', 'rs1', 'mic', 1, {'deactivated': False}),
#                 ServiceState(3, 'heartbeat', 'rs1', 'vfm', 1, {'deactivated': False}),
#                 ServiceState(4, 'heartbeat', 'rs1', 'vfm-geo', 1, {'deactivated': True}),
#                 ServiceState(5, 'heartbeat', 'rs1', 'vfm-mic', 1, {'deactivated': True}),
#                 ServiceState(6, 'noise_detector', 'rs1', 'nd', 1, {'mic': 100, 'geo': 1000}),
#             ],
#             'expected': [
#                 {'topic': 'rs1/vfm/ctl', 'payload': {'deactivated': True}},
#                 {'topic': 'rs1/vfm-mic/ctl', 'payload': {'deactivated': False}},
#             ],
#         },
#     }
#
#     for test_name in test_cases:
#         test_case = test_cases[test_name]
#         result = noise_check_node(ctl_topic, 'rs1', test_case['input'], deps)
#         assert result == test_case['expected'], result
