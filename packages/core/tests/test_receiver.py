# import json
# from collections import Counter
#
# import zenoh
#
# zenoh.init_logger()
# config = zenoh.Config()
# config.insert_json5(zenoh.config.MODE_KEY, json.dumps('peer'))
# config.insert_json5(
#     zenoh.config.CONNECT_KEY,
#     json.dumps(
#         [
#             'unixsock-stream///tmp/acies-topic_a.sock',
#             'unixsock-stream///tmp/acies-topic_b.sock',
#             'unixsock-stream///tmp/acies-topic_c.sock',
#         ]
#     ),
# )
# config.insert_json5(zenoh.config.LISTEN_KEY, json.dumps([]))
# # config.insert_json5('scouting/multicast/enabled', 'false')
#
# session = zenoh.open(config)
# msg_q = zenoh.Queue(bound=100)
#
#
# topics = ['topic_a', 'topic_b', 'topic_c']
# subs = {}
# for t in topics:
#     print(f'Subscribing to topic {t}')
#     subs[t] = session.declare_subscriber(t, msg_q, reliability=zenoh.Reliability.BEST_EFFORT())
#
# counter = Counter()
#
# try:
#     while True:
#         try:
#             sample = msg_q.get(timeout=1)
#             topic = str(sample.key_expr)
#             # payload = sample.payload
#             # msg = json.loads(payload)
#             # print(f'Received message at topic {topic}: {msg["timestamp"]}')
#             counter.update([topic])
#             print(dict(counter))
#         except (TimeoutError, StopIteration):
#             pass
# except KeyboardInterrupt:
#     print(counter)
#     pass
