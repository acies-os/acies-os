import ast
import cmd
import logging
import queue
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import click
import IPython
from acies.core import AciesMsg, common_options, get_zconf, init_logger
from acies.core import Service as _Service
from acies.core.service import pretty
from IPython.lib.pretty import pprint  # noqa: F401
from zenoh.closures import time

try:
    import readline
except ImportError:
    readline = None

acies_path = Path.home() / '.acies'
acies_path.mkdir(exist_ok=True)

histfile = acies_path / '.ash_history'
histfile_size = 1000


logger = logging.getLogger('acies')


class Service(_Service):

    def start(self):
        """Start the service.
        """
        self.start_time = datetime.now()
        self._ctl_thread = threading.Thread(target=self._handle_ctl_messages, args=(self.event,))
        self._ctl_thread.start()
        # self.schedule(self.heartbeat_interval_s, self._heartbeat, periodic=True)
        # self._scheduler.run()

    def shutdown(self):
        """Shutdown the service.
        """
        self._undeclare()
        # wait the thread in in _mb thread to respond to event set
        time.sleep(0.2)
        end_time = datetime.now()
        logger.info(
            f'{self.proc_name} finished in {(end_time-self.start_time).total_seconds()}s (from {self.start_time} to {end_time})'
        )


class AciesShell(cmd.Cmd):
    # intro = 'Type help or ? to list commands.\n'
    intro = 'hello!\n'
    prompt = 'Acies> '

    def __init__(self, service: Service, *args, **kwargs):
        """Initialize the interactive Acies shell (REPL).

        Starts the provided :class:`Service`, prints its control topic, and launches
        a background dispatch thread that drains :attr:`service.msg_q` into the
        in-memory :attr:`messages` buffer keyed by topic.

        Args:
            service: The running Acies :class:`Service` instance to bind to.
            *args: Forwarded to :class:`cmd.Cmd`.
            **kwargs: Forwarded to :class:`cmd.Cmd`.
        """

        super().__init__(*args, **kwargs)

        self.service = service
        self.nid = self.service.get_zid()

        self.event = threading.Event()
        self.messages = defaultdict(list)
        self.msg_count = 0

        self.service.start()
        # self.service.add_sub(self.service.ctrl_topic)
        print(self.service.ctrl_topic)

        self.dispatch_thread = threading.Thread(target=self.dispatch, args=(self.service.msg_q, self.event))
        self.dispatch_thread.daemon = True
        self.dispatch_thread.start()

        assert isinstance(self.intro, str)
        print(f'REPL id: {self.nid}')

    def emptyline(self):
        pass

    def update_prompt(self):
        """Update the prompt to reflect unseen messages.

        Sets the prompt to ``'Acies> '`` when there are no new messages, or to
        ``'Acies [ N new msgs ]> '`` when the message buffer has grown since the
        last :meth:`show`/print.
        """

        total = sum(len(x) for x in self.messages.values())
        delta = total - self.msg_count
        if delta == 0:
            self.prompt = 'Acies> '
        else:
            self.prompt = f'Acies [ {delta} new msgs ]> '

    def shutdown(self):
        """Shut down the REPL and its service.

        Signals the dispatch thread to stop, calls :meth:`Service.shutdown`, prints
        a goodbye message, and sleeps briefly to flush output.
        """

        # self.service._undeclare()
        self.service.shutdown()
        self.event.set()
        print('bye!')
        time.sleep(0.1)

    def dispatch(self, msg_q: queue.Queue, event: threading.Event):
        """Background loop that copies messages from the service queue.

        Reads ``(topic, AciesMsg)`` tuples from ``msg_q`` and appends them to the
        :attr:`messages` buffer until ``event`` is set.

        Args:
            msg_q: The service's application message queue.
            event: Stop signal for the dispatcher loop.
        """
        logger.info('start dispatch thread')
        while not event.is_set():
            try:
                topic, msg = msg_q.get(timeout=1)
                self.messages[topic].append(msg)
            except (queue.Empty, TimeoutError, StopIteration):
                pass
        logger.info('exiting dispatch thread')

    def postcmd(self, stop, line):
        """Hook after each command.

        Refreshes the dynamic prompt to include the new-message count.

        Args:
            stop: The value returned by the command handler.
            line: The raw input line.

        Returns:
            The unmodified ``stop`` value to control REPL exit flow.
        """

        self.update_prompt()
        return stop

    def preloop(self):
        """Load command history before entering the REPL loop.

        If ``readline`` is available and a history file exists, load it so that
        arrow-key history navigation works.
        """
        if readline and histfile.exists():
            readline.read_history_file(histfile)

    def postloop(self):
        """Persist command history and shut down after the REPL exits.

        Writes the history file (if ``readline`` is available) and then calls
        :meth:`shutdown`.
        """
        if readline:
            readline.set_history_length(histfile_size)
            readline.write_history_file(histfile)
        self.shutdown()

    def do_EOF(self, arg):
        """Exit the REPL on Ctrl-D (EOF).

        Args:
            arg: Unused.

        Returns:
            True to signal `cmd.Cmd` to exit.
        """

        return True

    def do_quit(self, arg):
        """Exit the REPL.

        Args:
            arg: Unused.

        Returns:
            True to signal `cmd.Cmd` to exit.
        """
        return True

    # def do_q(self, arg):
    #     return self.do_quit(arg)

    def _list(self):
        """Print peers and current subscriptions.

        Shows discovered peer ZIDs and the set of topics currently subscribed by
        the underlying :class:`Service`.
        """
        print('=== peer zids ===')
        for p in self.service.session.info().peers_zid():
            print(p)

        print('=== subscribing topics ===')
        for topic in self.service.active_subs.keys():
            print(topic)

    # def do_l(self, arg):
    #     self.do_list(arg)

    def do_list(self, arg):
        """List peers and subscribed topics (REPL command: ``list``).

        Args:
            arg: Unused.
        """
        self._list()

    def _print_msg(self, topic: str, msg: AciesMsg, full: bool):
        """Pretty-print a single message.

        Args:
            topic: The topic the message arrived on.
            msg: The message to display.
            full: If True, print the full dict; otherwise, print a compact preview.
        """
        msg_dict = msg.to_dict()
        if full:
            print(topic, '|', msg_dict)
        else:
            print(topic, '|', pretty(msg_dict, max_seq_length=6, max_width=500, newline=''))

    def do_ls(self, line):
        """Explore peers/topics like a directory tree (REPL command: ``ls``).

        Usage examples:
            - ``ls`` → prints roots ``topic/`` and ``peer/``.
            - ``ls peer`` → lists peer ZIDs.
            - ``ls topic/<parts>`` → when a recorded topic matches:
                * prints hosts (unique ``reply_to`` values), or
                * for a host, prints timestamps of stored messages.

        Args:
            line: Optional path after ``ls``, such as ``peer`` or ``topic/...``.
        """

        parts = line.split('/')
        if parts == ['']:
            print('topic/\npeer/')
        elif parts == ['peer']:
            for p in self.service.session.info().peers_zid():
                print(f'peer/{p}')
        elif line.startswith('topic'):
            arg = line.removeprefix('topic').strip('/')
            parts = arg.split('/')
            for k in range(len(parts)):
                part1 = '/'.join(parts[: k + 1])
                part2 = '/'.join(parts[k + 1 :])
                if part1 in self.messages:
                    if part2 == '':
                        hosts = list(sorted(set(x.reply_to for x in self.messages[part1])))
                        for h in hosts:
                            print(f'topic/{part1}/{h}')
                    else:
                        timestamps = list(sorted(set(x.timestamp for x in self.messages[part1] if x.reply_to == part2)))
                        for t in timestamps:
                            print(f'topic/{part1}/{part2}/{t}')
                    return
            else:
                for topic in self.messages.keys():
                    print(f'topic/{topic}')

    def do_cat(self, line):
        """Print a single recorded message by path (REPL command: ``cat``).

        Usage:
            ``cat topic/<topic>/<reply_to>/<timestamp>``

        Looks up the message with the given topic, reply target, and timestamp and
        prints it in a human-readable form.

        Args:
            line: The path specifying which message to show.
        """

        arg = line.removeprefix('topic').strip('/')
        parts = arg.split('/')
        timestamp = parts[-1]
        parts = tuple(parts[:-1])
        for k in range(len(parts)):
            part1 = '/'.join(parts[: k + 1])
            part2 = '/'.join(parts[k + 1 :])
            if part1 in self.messages:
                for msg in self.messages[part1]:
                    if msg.reply_to == part2 and msg.timestamp == int(timestamp):
                        msg_dict = msg.to_dict()
                        print(pretty(msg_dict, max_seq_length=6, max_width=500, newline=''))
                        return
        print('invalid args')

    def show(self, target: str, full: bool):
        """Display messages from the buffer.

        Args:
            target: One of ``'topics'`` (list subscriptions and known topics),
                ``'all'`` (dump all messages), or a specific topic string.
            full: If True, print each message as a full dict; else a compact form.

        Notes:
            Resets the “new messages” counter after printing relevant messages.
        """
        
        # print messages in the selected topic
        if target in self.messages:
            for msg in self.messages[target]:
                self._print_msg(target, msg, full)

        # list all topics
        elif target == 'topics':
            print('=== subs ===')
            for topic in self.service.active_subs:
                print(topic)
            print('=== topics ===')
            for topic in self.messages.keys():
                print(topic)

        # show all messages
        elif target == 'all':
            for topic, msgs in self.messages.items():
                print(f'=== {topic} ===')
                for msg in msgs:
                    self._print_msg(topic, msg, full)

        # unknown target
        else:
            print(f'unknown target: {target}')
            print('available target: "topics", "all", or a specific topic')
            # early return to avoid updating msg_count
            return

        # clear new message notification
        self.msg_count = sum(len(x) for x in self.messages.values())

    def do_clear(self, arg):
        """Clear the in-memory message buffer (REPL command: ``clear``).

        Args:
            arg: Unused.
        """

        self.messages.clear()
        self.msg_count = 0

    def do_show(self, arg):
        """Wrapper around :meth:`show` (REPL command: ``show``).

        Usage:
            ``show all`` |
            ``show topics`` |
            ``show <topic> [full]``

        Args:
            arg: Target and optional ``full`` flag.

        Returns:
            Optional usage string if the arguments are malformed (printed by `cmd.Cmd`).
        """
        args = arg.split(' ', 1)
        if len(args) == 0:
            return 'usage: show all|topics|<specific/topic> [full]'
        elif len(args) == 1:
            self.show(args[0], False)
        else:
            self.show(args[0], True)

    def send(self, topic: str, msg_type: str, payload: dict | list | None, meta: dict | None):
        """Construct and publish a message via the bound service.

        Args:
            topic: Topic to publish to.
            msg_type: Message kind (e.g., ``'json'``, ``'get'``, ``'set'``,
                ``'topic'``, ``'reply'``, or ``'array_<dtype>'``).
            payload: Message payload (type depends on ``msg_type``).
            meta: Optional metadata to attach.

        Raises:
            ValueError: Propagated if the payload type is invalid for the given kind.
        """
        try:
            msg = self.service.make_msg(msg_type, payload, meta)
            self.service.send(topic, msg)
        except ValueError as e:
            print(f'malformed input {e}: {msg_type=}, {payload=}, {meta=}')

    @staticmethod
    def _suffix_ctl(topic: str) -> str:
        """Ensure a control suffix on a topic path.

        Appends ``'/ctl'`` to ``topic`` if it is not already present.

        Args:
            topic: Base topic string.

        Returns:
            The control topic string.
        """
        if not topic.endswith('/ctl'):
            return topic + '/ctl'
        return topic

    def do_param_set(self, arg):
        """Send a control ``set`` to a service (REPL command: ``param_set``).

        Usage:
            ``param_set <topic> {"key": value, ...}``

        Parses the dictionary and sends a ``set`` control message to
        ``<topic>/ctl`` with the provided key/value pairs.

        Args:
            arg: Topic and dict string.
        """


        try:
            topic, val = arg.split(' ', 1)
            val = ast.literal_eval(val)
            assert isinstance(val, dict)
        except ValueError:
            print('usage: param_set <topic> <dict[key, val]>')
            return

        topic = self._suffix_ctl(topic)
        self.send(topic, 'set', val, None)

    def do_param_get(self, arg):
        """Send a control ``get`` to a service (REPL command: ``param_get``).

        Usage:
            ``param_get <topic> <key1> <key2> ...``

        Requests the specified keys (or entire state if you pass ``*``) from
        ``<topic>/ctl`` and queues the reply in the message buffer.

        Args:
            arg: Topic and one or more keys separated by spaces.
        """

        try:
            topic, keys = tuple(arg.split(' ', 1))
            keys = keys.split(' ')
            print(topic)
            print(keys)
        except ValueError:
            print('usage: param_get <topic> <keys separated by spaced>')
            return

        topic = self._suffix_ctl(topic)
        self.send(topic, 'get', list(keys), None)

    def do_activate(self, arg):
        """Activate a service by clearing its ``deactivated`` flag (REPL: ``activate``).

        Sends ``{'deactivated': False}`` to ``<topic>/ctl``.

        Args:
            arg: Topic string (without or with ``/ctl`` suffix).
        """

        arg = self._suffix_ctl(arg)
        self.send(arg, 'set', {'deactivated': False}, {})

    def do_deactivate(self, arg):
        """Deactivate a service by setting its ``deactivated`` flag (REPL: ``deactivate``).

        Sends ``{'deactivated': True}`` to ``<topic>/ctl``.

        Args:
            arg: Topic string (without or with ``/ctl`` suffix).
        """

        arg = self._suffix_ctl(arg)
        self.send(arg, 'set', {'deactivated': True}, {})

    def do_down(self, arg):
        """Disable heartbeat emission (REPL command: ``down``).

        Sends ``{'enable_heartbeat': False}`` to ``<topic>/ctl``.

        Args:
            arg: Topic string (without or with ``/ctl`` suffix).
        """
        arg = self._suffix_ctl(arg)
        self.send(arg, 'set', {'enable_heartbeat': False}, {})

    def do_up(self, arg):
        """Enable heartbeat emission (REPL command: ``up``).

        Sends ``{'enable_heartbeat': True}`` to ``<topic>/ctl``.

        Args:
            arg: Topic string (without or with ``/ctl`` suffix).
        """
        arg = self._suffix_ctl(arg)
        self.send(arg, 'set', {'enable_heartbeat': True}, {})

    def do_send(self, arg):
        """Send an arbitrary message (REPL command: ``send``).

        Usage:
            ``send <topic> <msg_type> {"payload": {...}, "meta": {...}}``

        Parses a JSON-like dict (via ``ast.literal_eval``), extracts ``payload`` and
        ``meta``, constructs an :class:`AciesMsg`, and publishes it.

        Args:
            arg: Topic, message kind, and content dict string.
        """

        try:
            topic, msg_type, content = tuple(arg.split(' ', 2))
        except ValueError:
            print('usage: send <topic> <msg_type> <content: dict>')
            print('example:')
            print('send ns/name/ctl data {"payload": {"hello": "world"}, "meta": {"key": 1.0}}')
            return

        try:
            content = ast.literal_eval(content)
            assert isinstance(content, dict)
            payload = content.pop('payload', None)
            meta = content.pop('meta', None)
            self.send(topic, msg_type, payload, meta)
        except (SyntaxError, ValueError, KeyError, TypeError, AssertionError) as e:
            print(f'malformed input: {content}, {e}')


@click.command()
@common_options
@click.option(
    '-i',
    '--ipython',
    is_flag=True,
    default=False,
    show_default=True,
    help='Drop into ipython repl instead of the shell.',
)
def main(mode, topic, connect, listen, ipython, proc_name, namespace, deactivated):
    """Entry point for the Acies shell CLI.

    Initializes logging, builds a Zenoh configuration, starts a minimal
    :class:`Service` (no heartbeat), subscribes to any ``--topic`` values, and
    launches either the interactive REPL or an IPython session.

    Args:
        mode: Zenoh mode (e.g., ``"peer"``, ``"client"``, ``"router"``) or None.
        topic: One or more topics to subscribe to at startup (multiple flags).
        connect: Endpoint(s) to connect to, or None.
        listen: Endpoint(s) to listen on, or None.
        ipython: If True, drop into IPython instead of the shell.
        proc_name: Process name used for logs/control topic (defaults to ``"ash"``).
        namespace: Optional namespace prefix for topic construction.
        deactivated: If True, start with message handling deactivated.
    """

    init_logger(f'ash_{namespace}.log')

    if proc_name is None:
        proc_name = 'ash'

    # zenoh config
    zconf = get_zconf(mode, connect, listen)

    # a middleware object used to send/receive messages
    service = Service(zconf, namespace=namespace, proc_name=proc_name, deactivated=deactivated, enable_heartbeat=False)

    # Add subscription topics
    for k in topic:
        service.add_sub(k)

    ashell = AciesShell(service)

    if ipython:
        # with this flag, the program drops you in a ipython shell, so that you
        # can interact with the system programmatically, easier to debug.
        IPython.embed()
        ashell.shutdown()
    else:
        try:
            time.sleep(0.1)
            ashell.cmdloop()
        except KeyboardInterrupt:
            ashell.shutdown()
