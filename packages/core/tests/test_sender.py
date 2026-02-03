# import json
# import sys
# import time
#
# import zenoh
#
# topic = sys.argv[1]
#
# zenoh.init_logger()
# config = zenoh.Config()
# config.insert_json5(zenoh.config.MODE_KEY, json.dumps('peer'))
# config.insert_json5(zenoh.config.CONNECT_KEY, json.dumps([]))
# config.insert_json5(
#     zenoh.config.LISTEN_KEY,
#     json.dumps([f'unixsock-stream///tmp/acies-{topic}.sock']),
# )
# # config.insert_json5('scouting/multicast/enabled', 'false')
#
# session = zenoh.open(config)
#
# print(topic)
# counter = 0
# N = 16000
#
# try:
#     while True:
#         msg = {
#             'msg_type': 'data',
#             'timestamp': time.time_ns(),
#             'payload': list(range(N)),
#             'metadata': {'channel': 0, 'sample_rate': N},
#         }
#         msg = json.dumps(msg)
#         session.put(topic, msg, congestion_control=zenoh.CongestionControl.DROP())
#         counter += 1
#         print(f'{topic}, {len(msg)} bytes: {counter}')
#         time.sleep(1)
# except KeyboardInterrupt:
#     print(f'sent {counter} messages to topic {topic}')
#     pass
