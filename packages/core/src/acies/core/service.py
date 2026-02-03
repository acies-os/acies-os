import json
import logging
import pickle
import queue
import sched
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional

import psutil
import zenoh
from acies.core.types import AciesMsg
from IPython.lib.pretty import pretty  # noqa: F401

logger = logging.getLogger('acies.service')


def get_cpu_avg_load():
    """Get the average CPU load.

    Returns:
        dict[str, float]: A dictionary containing the CPU load averages for 1, 5, and 15 minutes.
    """
    return dict(zip(['cpu_1min', 'cpu_5min', 'cpu_15min'], psutil.getloadavg()))


def get_mem_info():
    """Get memory information.

    Returns:
        dict[str, float]: A dictionary containing memory information.
    """
    info = psutil.virtual_memory()
    info = {
        'mem_total_bytes': info.total,
        'mem_available_bytes': info.available,
        'mem_percentage': (info.total - info.available) / (info.total),
    }
    return info


def get_sys_info():
    """Get system information.

    Returns:
        dict[str, float]: A dictionary containing CPU and memory information.
    """
    cpu_info = get_cpu_avg_load()
    mem_info = get_mem_info()
    return {**cpu_info, **mem_info}


def get_zconf(mode, connect, listen) -> zenoh.Config:
    """Build a Zenoh configuration from optional CLI-like arguments.

    Args:
        mode: Zenoh mode value, or None.
        connect: Endpoint(s) to connect to, or None.
        listen: Endpoint(s) to listen on, or None.

    Returns:
        zenoh.Config: Populated configuration object.
    """

    zenoh.init_logger()
    conf = zenoh.Config()
    if mode is not None:
        conf.insert_json5(zenoh.config.MODE_KEY, json.dumps(mode))
    if connect is not None:
        conf.insert_json5(zenoh.config.CONNECT_KEY, json.dumps(connect))
    if listen is not None:
        conf.insert_json5(zenoh.config.LISTEN_KEY, json.dumps(listen))
    return conf


class Service:
    """Base class for long-running Acies services over Zenoh.

    This class wraps Zenoh session management, pub/sub registration, control-message
    routing, a lightweight scheduler for periodic/one-shot tasks, and liveness/
    diagnostic heartbeats. Subclasses implement :meth:`run` to register work
    (e.g., schedule tasks or consume ``msg_q``) and call :meth:`start` to launch
    the control thread and enter the scheduler loop.

    Args:
        conf (zenoh.Config): Configuration used to open the Zenoh session.
        namespace (str | None, optional): Namespace prefix applied by
            :meth:`ns_topic_str` when constructing topics.
        proc_name (str, optional): Process name used in logs and in the default
            control topic when ``ctrl_topic`` is omitted. Defaults to the Zenoh ZID.
        ctrl_topic (str, optional): Control topic for ``get``/``set``/``topic``/``reply``
            messages. Defaults to ``ns_topic_str(proc_name, "ctl")``.
        topic (Iterable[str], optional): Additional topics to subscribe to at start.
        deactivated (bool, optional): If ``True``, application messages are dropped.
        enable_heartbeat (bool, optional): If ``True``, emit heartbeat messages.
        heartbeat_interval_s (int, optional): Seconds between heartbeats. Defaults to 1.
        diagnostic_interval_s (int, optional): Minimum seconds between diagnostic
            payloads included with heartbeats. Defaults to 3.

    Attributes:
        session (zenoh.Session): Active Zenoh session.
        active_subs (dict[str, zenoh.Subscriber]): Current subscriptions (thread-safe).
        active_pubs (dict[str, zenoh.Publisher]): Lazily created publishers (thread-safe).
        msg_q (queue.Queue[tuple[str, AciesMsg]]): Queue of application messages
            forwarded from the control thread.
        service_states (dict): Mutable service state/parameters. Includes keys like
            ``"deactivated"`` and ``"enable_heartbeat"``, and any values set via
            control messages.
        namespace (str | None): Namespace used by :meth:`ns_topic_str`.
        proc_name (str): Process identifier used in topics and logs.
        ctrl_topic (str): Control topic for this service.
        heartbeat_interval_s (int): Heartbeat period in seconds.
        diagnostic_interval_s (datetime.timedelta): Interval gating diagnostic payloads.
        last_diagnostic (datetime.datetime): Timestamp of the last diagnostic emission.
        event (threading.Event): Stop signal for background control processing.
        _scheduler (sched.scheduler): Scheduler for periodic/one-shot tasks.

    Message flow:
        Incoming Zenoh samples are delivered to an internal queue and processed
        by a background control thread. Control messages of kind ``"set"``,
        ``"get"``, ``"topic"``, and ``"reply"`` are handled internally; all other
        messages are forwarded to :attr:`msg_q` for application logic.
        Use :meth:`make_msg` / :meth:`make_reply` to construct messages and
        :meth:`send` to publish them. Supported kinds include:
        ``array_i16``, ``array_i32``, ``array_i64``, ``array_f64``, ``json``,
        ``heartbeat``, ``get``, ``set``, ``topic``, and ``reply``.

    Lifecycle:
        * Call :meth:`start` to spawn the control thread, schedule heartbeats,
          invoke :meth:`run`, and enter the scheduler loop.
        * Override :meth:`run` in subclasses to register work (e.g., via
          :meth:`schedule`) or to consume :attr:`msg_q`.
        * Override :meth:`shutdown` for application-specific cleanup. Internal
          resources (subs/pubs/session) are released by the framework via
          :meth:`_undeclare`.

    Thread-safety:
        Subscription and publisher maps are protected by locks. :meth:`send`
        creates publishers on first use. Use :meth:`remove_sub` / :meth:`remove_pub`
        or rely on shutdown to release resources.
    """
    def __init__(self, conf: zenoh.Config, *args, **kwargs):
        self.session = zenoh.open(conf)
        # self.active_subs: dict[str, tuple[threading.Event, threading.Thread]] = {}
        self.active_subs: dict[str, zenoh.Subscriber] = {}
        self.active_subs_lock = threading.Lock()
        self.active_pubs: dict[str, zenoh.Publisher] = {}
        self.active_pubs_lock = threading.Lock()
        self._inner_q: queue.Queue[tuple[str, AciesMsg]] = queue.Queue()
        self.msg_q: queue.Queue[tuple[str, AciesMsg]] = queue.Queue()

        self.event = threading.Event()
        self._scheduler = sched.scheduler()

        self.namespace: str | None = kwargs.get('namespace', None)
        self.proc_name: str = kwargs.get('proc_name', self.get_zid())
        self.ctrl_topic: str = kwargs.get('ctrl_topic', self.ns_topic_str(self.proc_name, 'ctl'))
        assert self.ctrl_topic is not None

        self._service_states = {
            'deactivated': kwargs.get('deactivated', False),
            'enable_heartbeat': kwargs.get('enable_heartbeat', True),
        }

        self.add_sub(self.ctrl_topic)
        for topic in kwargs.get('topic', []):
            self.add_sub(topic)

        now = datetime.now()

        self.heartbeat_interval_s = int(kwargs.get('heartbeat_interval_s', 1))

        self.diagnostic_interval_s = timedelta(seconds=kwargs.get('diagnostic_interval_s', 3))
        self.last_diagnostic = now - self.diagnostic_interval_s

    def schedule(
        self,
        delay: float,
        func: Callable,
        priority: Any = 1,
        periodic: bool = False,
        absolute_time: bool = False,
        func_args: tuple = (),
    ) -> sched.Event:
        """Schedule a function to run after a delay. If periodic is True, the function will be called repeatedly    
        Args:
            delay: Seconds until the call (or absolute timestamp if ``absolute_time=True``).
            func: Callable to execute.
            priority: Lower values run first if multiple events share the same time.
            periodic: If True, reschedules itself with the same delay after each run.
            absolute_time: If True, ``delay`` is treated as an absolute timestamp.

            func_args: Positional arguments to pass to ``func``.

        Returns:
            sched.Event: The scheduled event (can be canceled).
        """

        sched_func = self._scheduler.enterabs if absolute_time else self._scheduler.enter

        if periodic:
            # schedule the next call before calling the function
            event = sched_func(
                delay,
                priority,
                self.schedule,
                (delay, func, priority, periodic, func_args),
            )
            # call the function
            func(*func_args)
        else:
            # schedule a one-time call
            event = sched_func(delay, priority, func, func_args)

        # return the event object which can be used to cancel the scheduled call
        return event

    def _cancel_events(self):
        for e in self._scheduler.queue:
            try:
                self._scheduler.cancel(e)
            except ValueError:
                pass

    def make_reply(self, req_msg: AciesMsg, reply_payload: dict, timestamp_ns: int | None = None) -> AciesMsg:
        """Create a reply control message for a prior request.

        Args:
            req_msg: The original request message being answered.
            reply_payload: Arbitrary reply payload to embed under ``"respond"``.
            timestamp_ns: Optional nanosecond timestamp to stamp on the message.

        Returns:
            AciesMsg: A control message of kind ``"reply"`` addressed to
            :attr:`ctrl_topic` containing the response and request metadata.
        """

        payload = {
            'respond': reply_payload,
            'request_timestamp': req_msg.timestamp,
            'request_type': req_msg.kind,
        }
        msg = self.make_msg(
            'reply',
            payload=payload,
            reply_to=self.ctrl_topic,
            timestamp_ns=timestamp_ns,
        )
        return msg

    def make_msg(
        self,
        kind: str,
        payload: Any,
        meta: Optional[Dict[str, Any]] = None,
        reply_to: Optional[str] = None,
        timestamp_ns: int | None = None,
    ) -> AciesMsg:
        """Construct an :class:`AciesMsg` of a given kind.

        Supported kinds:
            - ``"array_<dtype>"``: payload must be ``list``; ``dtype`` in {``i16``, ``i32``, ``i64``, ``f64``}.
            - ``"json"``: payload must be ``dict``.
            - ``"heartbeat"``: no payload requirements.
            - Control kinds: ``"get"``, ``"set"``, ``"topic"``, ``"reply"``.

        Args:
            kind: Message kind selector (see above).
            payload: Message payload (type depends on ``kind``).
            meta: Optional metadata dict to attach.
            reply_to: Topic to address; defaults to :attr:`ctrl_topic` if ``None``.
            timestamp_ns: Optional nanosecond timestamp for the message.

        Returns:
            AciesMsg: The constructed message.

        Raises:
            AssertionError: If payload type does not match the selected ``kind``.
            ValueError: If ``kind`` is not one of the supported kinds.
        """

        # check reply_to
        if reply_to is None:
            reply_to = self.ctrl_topic
        assert reply_to is not None

        # # check meta
        # if meta is None:
        #     meta = {}
        # assert isinstance(meta, dict)

        if kind.startswith('array_'):
            data_type = kind.split('_')[1]
            assert isinstance(payload, list)
            msg = AciesMsg.new_array_msg(payload, reply_to, meta, data_type=data_type, timestamp_ns=timestamp_ns)
        elif kind == 'json':
            assert isinstance(payload, dict)
            msg = AciesMsg.new_json_msg(reply_to, payload, meta, timestamp_ns=timestamp_ns)
        elif kind == 'heartbeat':
            msg = AciesMsg.new_heartbeat(reply_to, meta, timestamp_ns=timestamp_ns)
        elif kind in ['get', 'set', 'topic', 'reply']:
            msg = AciesMsg.new_ctl_msg(kind, reply_to, payload, meta, timestamp_ns)
        else:
            raise ValueError(
                f'Invalid kind: {kind}, currently supported: array_i16, array_i32, array_i64, array_f64, json, heartbeat, get, set, topic, reply'
            )
        assert reply_to is not None
        return msg

    @property
    def service_states(self):
        """dict: Mutable service state dictionary.

        Holds configuration/state values (e.g., ``"deactivated"``,
        ``"enable_heartbeat"``) and any keys set via control messages.
        """

        return self._service_states

    @service_states.setter
    def service_states(self, param_dict: Dict):

        assert isinstance(param_dict, dict)
        self._service_states = param_dict

    def topic_str(self, *args) -> str:
        """Build a topic string by joining components with ``'/'``.

        ``None`` components are skipped.

        Args:
            *args: Topic path segments.

        Returns:
            str: The joined topic string.
        """

        return '/'.join(x for x in args if x is not None)

    def ns_topic_str(self, *args) -> str:
        return self.topic_str(self.namespace, *args)

    def get_zid(self) -> str:
        """Return the Zenoh session ZID as a string

        Returns:
            str: The ZID as a string.
        """
        return str(self.session.info().zid())

    def send(self, topic: str, msg: AciesMsg):
        """Send a message to a specific topic.

        Args:
            topic (str): The topic to send the message to.
            msg (AciesMsg): The message to send.
        """
        try:
            pub = self.active_pubs[topic]
        except KeyError:
            pub = self.session.declare_publisher(topic)
            self.active_pubs[topic] = pub
        data = msg.to_bytes()
        pub.put(data)
        log_msg = {'send_msg_to': topic, 'bytes': len(data)}
        logger.debug(f'{log_msg}')

    # def _sub_thread(self, topic: str, stop_event: threading.Event):
    #     sub = self.session.declare_subscriber(
    #         topic,
    #         zenoh.Queue(bound=100),
    #         reliability=zenoh.Reliability.BEST_EFFORT(),
    #     )
    #     assert sub.receiver is not None
    #     try:
    #         for sample in sub.receiver:
    #             if stop_event.is_set():
    #                 break
    #             topic = str(sample.key_expr)
    #             msg = AciesMsg.from_bytes(sample.payload)
    #             self.msg_q.put((topic, msg))
    #     finally:
    #         sub.undeclare()
    #         logger.debug(f'Unsubscribed from {topic}')

    # def add_sub(self, topic: str):
    #     if topic in self.active_subs:
    #         return
    #     stop_event = threading.Event()
    #     thread = threading.Thread(target=self._sub_thread, args=(topic, stop_event))
    #     thread.start()
    #     logger.debug(f'Subscribed to key "{topic}"')
    #     with self.active_subs_lock:
    #         self.active_subs[topic] = (stop_event, thread)

    def _sub_handle(self, sample):
        topic = str(sample.key_expr)
        msg = AciesMsg.from_bytes(sample.payload)
        self._inner_q.put((topic, msg))

    def add_sub(self, topic: str):
        """Subscribe to a topic if not already subscribed.

        Declares a Zenoh subscriber with best-effort reliability and
        registers the internal handler.


        Args:
            topic (str): The topic to subscribe to.
        """
        if topic in self.active_subs:
            return
        sub = self.session.declare_subscriber(
            topic,
            handler=self._sub_handle,
            reliability=zenoh.Reliability.BEST_EFFORT(),
        )
        with self.active_subs_lock:
            self.active_subs[topic] = sub

    # def remove_sub(self, topic: str):
    #     try:
    #         stop_event, thread = self.active_subs[topic]
    #         stop_event.set()
    #         thread.join()
    #         with self.active_subs_lock:
    #             del self.active_subs[topic]
    #         logger.debug(f'removed {topic}')
    #     except KeyError:
    #         logger.warning(f'key "{topic}" not found in active_subs')

    def remove_sub(self, topic: str):
        """Remove a subscription to a topic.

        Args:
            topic (str): The topic to unsubscribe from.
        """
        try:
            sub = self.active_subs[topic]
            sub.undeclare()
            with self.active_subs_lock:
                del self.active_subs[topic]
        except KeyError:
            logger.warning(f'key "{topic}" not found in active_subs')

    def remove_pub(self, topic: str):
        """Undeclare and remove a cached publisher for a topic.

        Args:
            topic (str): Topic whose publisher should be removed.
        """
        try:
            pub = self.active_pubs.pop(topic)
            assert isinstance(pub, zenoh.Publisher)
            pub.undeclare()
            del pub
            logger.debug(f'removed {topic}')
        except KeyError:
            logger.warning(f'key "{topic}" not found in active_pubs')

    def _undeclare(self):
        self.event.set()
        # stop scheduler and scheduled events
        self._cancel_events()
        for topic in list(self.active_subs.keys()):
            self.remove_sub(topic)
        for topic in list(self.active_pubs.keys()):
            self.remove_pub(topic)
        self._ctl_thread.join()
        logger.debug('_handle_ctl_messages thread joined')
        self.session.close()
        logger.debug('session closed')

    @staticmethod
    def encode_payload(data) -> bytes:
        """Serialize a payload to bytes for transport.

        Uses JSON (UTF-8) by default; raise ``ValueError`` if not serializable.

        Args:
            data: JSON-serializable Python object.

        Returns:
            bytes: Serialized payload.

        Raises:
            ValueError: If ``data`` cannot be JSON-encoded.
        """

        try:
            # msg = pickle.dumps(msg)
            msg = json.dumps(data)
        except ValueError:
            raise ValueError(f'cannot encode msg: {data}')
        assert isinstance(msg, bytes)
        return msg

    @staticmethod
    def decode_payload(data):
        """Deserialize payload bytes to a Python object.

        Tries pickle first, then JSON (UTF-8). Raises on failure.

        Args:
            data: Raw payload bytes.

        Returns:
            Any: Decoded Python object.

        Raises:
            ValueError: If the data cannot be decoded as pickle or JSON.
        """

        try:
            # try to deserialize with pickle
            msg = pickle.loads(data)
        except (pickle.UnpicklingError, AttributeError, EOFError):
            try:
                # try to deserialize with json
                msg = json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ValueError(f'Failed to deserialize data using pickle or json: {data}')

        return msg

    def _handle_param_set(self, topic: str, msg: AciesMsg):
        payload = msg.get_payload()
        if not isinstance(payload, dict):
            logger.warning(f'unhandled param_set payload has to be a dict; got {type(payload)}: {topic=}, {msg=}')
            return
        for k, v in payload.items():
            logger.debug(f'set {k} from {self.service_states.get(k, "not found")} to {v}')
            self.service_states[k] = v
        reply = self.make_reply(msg, {k: self.service_states.get(k) for k in payload.keys()})
        return_topic = msg.reply_to
        self.send(return_topic, reply)

    def _handle_param_get(self, topic: str, msg: AciesMsg):
        payload = msg.get_payload()
        if (
            isinstance(payload, dict)
            or isinstance(payload, list)
            or isinstance(payload, tuple)
            or isinstance(payload, set)
        ):
            reply_payload = {k: self.service_states.get(k, 'not found') for k in payload}
        elif isinstance(payload, str):
            if payload == '*':
                reply_payload = {k: v for k, v in self.service_states.items()}
            else:
                reply_payload = {payload: self.service_states.get(payload, 'not found')}
        else:
            logger.warning(f'unhandled param_set payload type {type(payload)}: {topic=}, {msg=}')
            return
        reply = self.make_reply(msg, reply_payload)
        return_topic = msg.reply_to
        self.send(return_topic, reply)

    def _handle_topic_ctl(self, _topic: str, msg: AciesMsg):
        payload = msg.get_payload()
        assert isinstance(payload, dict)
        for topic, action in payload.items():
            if action == 'add':
                self.add_sub(topic)
            elif action == 'remove':
                self.remove_sub(topic)
            else:
                logger.warning(f'unhandled topic_ctl action: {action}')

    def _handle_reply(self, topic: str, msg: AciesMsg):
        logger.debug(f'received reply: {msg}')
        self.msg_q.put((topic, msg))

    # def _get_msg(self, timeout: float | None = None) -> tuple[str, AciesMsg]:
    #     sample: zenoh.Sample = self._inner_q.get(timeout)  # type: ignore
    #     topic = str(sample.key_expr)
    #     msg = AciesMsg.from_bytes(sample.payload)
    #     return (topic, msg)

    def _handle_ctl_messages(self, event: threading.Event):
        logger.debug('starting _handle_ctl_messages')
        while not event.is_set():
            try:
                (topic, msg) = self._inner_q.get(block=True, timeout=1)
                if msg.kind == 'set':
                    self._handle_param_set(topic, msg)
                elif msg.kind == 'get':
                    self._handle_param_get(topic, msg)
                elif msg.kind == 'topic':
                    self._handle_topic_ctl(topic, msg)
                elif msg.kind == 'reply':
                    self._handle_reply(topic, msg)
                else:
                    # forward to the application
                    self.msg_q.put((topic, msg))
            except (TimeoutError, StopIteration, queue.Empty):
                pass
        logger.debug('exiting _handle_ctl_messages')

    def _heartbeat(self):
        now = datetime.now()
        if now - self.last_diagnostic >= self.diagnostic_interval_s:
            payload = get_sys_info()
            self.last_diagnostic = now
        else:
            payload = {}
        meta = {
            'deactivated': self.service_states.get('deactivated'),
        }
        if self._service_states.get('enable_heartbeat', False):
            msg = self.make_msg('heartbeat', payload, meta)
            self.send('heartbeat', msg)

    def shutdown(self):
        pass

    def run(self):
        raise NotImplementedError('This method should be overridden by the child class')

    def start(self):
        """Start the service control thread, schedule heartbeats, and enter the scheduler loop.

        Spawns the control-message thread, sets up periodic heartbeat emission,
        invokes :meth:`run`, and then runs the scheduler until interrupted.
        Ensures shutdown and resource cleanup in a ``finally`` block.
        """

        logger.info(f'service {self.ctrl_topic} starts: deactivated={self.service_states["deactivated"]}')
        start_time = datetime.now()
        try:
            self._ctl_thread = threading.Thread(target=self._handle_ctl_messages, args=(self.event,))
            self._ctl_thread.start()
            self.schedule(self.heartbeat_interval_s, self._heartbeat, periodic=True)
            self.run()
            self._scheduler.run()
        except KeyboardInterrupt:
            return
        finally:
            # self._cancel_events()
            self.shutdown()
            self._undeclare()
            # wait the subscriber thread to respond to event set
            time.sleep(0.2)
            end_time = datetime.now()
            logger.info(
                f'{self.proc_name} finished in {(end_time-start_time).total_seconds()}s (from {start_time} to {end_time})'
            )
