# acies.core.shell

### Classes

| [`AciesShell`](#acies.core.shell.AciesShell)(service, \*args, \*\*kwargs)   |    |
|-----------------------------------------------------------------------------|----|
| [`Service`](#acies.core.shell.Service)(conf, \*args, \*\*kwargs)            |    |

### *class* acies.core.shell.AciesShell(service, \*args, \*\*kwargs)

Bases: [`Cmd`](https://docs.python.org/3/library/cmd.html#cmd.Cmd)

* **Parameters:**
  **service** ([`Service`](#acies.core.shell.Service))

#### \_\_init_\_(service, \*args, \*\*kwargs)

Initialize the interactive Acies shell (REPL).

Starts the provided [`Service`](#acies.core.shell.Service), prints its control topic, and launches
a background dispatch thread that drains `service.msg_q` into the
in-memory `messages` buffer keyed by topic.

* **Parameters:**
  * **service** ([`Service`](#acies.core.shell.Service)) – The running Acies [`Service`](#acies.core.shell.Service) instance to bind to.
  * **\*args** – Forwarded to [`cmd.Cmd`](https://docs.python.org/3/library/cmd.html#cmd.Cmd).
  * **\*\*kwargs** – Forwarded to [`cmd.Cmd`](https://docs.python.org/3/library/cmd.html#cmd.Cmd).

#### dispatch(msg_q, event)

Background loop that copies messages from the service queue.

Reads `(topic, AciesMsg)` tuples from `msg_q` and appends them to the
`messages` buffer until `event` is set.

* **Parameters:**
  * **msg_q** ([`Queue`](https://docs.python.org/3/library/queue.html#queue.Queue)) – The service’s application message queue.
  * **event** ([`Event`](https://docs.python.org/3/library/threading.html#threading.Event)) – Stop signal for the dispatcher loop.

#### do_EOF(arg)

Exit the REPL on Ctrl-D (EOF).

* **Parameters:**
  **arg** – Unused.
* **Returns:**
  True to signal cmd.Cmd to exit.

#### do_activate(arg)

Activate a service by clearing its `deactivated` flag (REPL: `activate`).

Sends `{'deactivated': False}` to `<topic>/ctl`.

* **Parameters:**
  **arg** – Topic string (without or with `/ctl` suffix).

#### do_cat(line)

Print a single recorded message by path (REPL command: `cat`).

Usage:
: `cat topic/<topic>/<reply_to>/<timestamp>`

Looks up the message with the given topic, reply target, and timestamp and
prints it in a human-readable form.

* **Parameters:**
  **line** – The path specifying which message to show.

#### do_clear(arg)

Clear the in-memory message buffer (REPL command: `clear`).

* **Parameters:**
  **arg** – Unused.

#### do_deactivate(arg)

Deactivate a service by setting its `deactivated` flag (REPL: `deactivate`).

Sends `{'deactivated': True}` to `<topic>/ctl`.

* **Parameters:**
  **arg** – Topic string (without or with `/ctl` suffix).

#### do_down(arg)

Disable heartbeat emission (REPL command: `down`).

Sends `{'enable_heartbeat': False}` to `<topic>/ctl`.

* **Parameters:**
  **arg** – Topic string (without or with `/ctl` suffix).

#### do_list(arg)

List peers and subscribed topics (REPL command: `list`).

* **Parameters:**
  **arg** – Unused.

#### do_ls(line)

Explore peers/topics like a directory tree (REPL command: `ls`).

Usage examples:
: - `ls` → prints roots `topic/` and `peer/`.
  - `ls peer` → lists peer ZIDs.
  - `ls topic/<parts>` → when a recorded topic matches:
    : * prints hosts (unique `reply_to` values), or
      * for a host, prints timestamps of stored messages.

* **Parameters:**
  **line** – Optional path after `ls`, such as `peer` or `topic/...`.

#### do_param_get(arg)

Send a control `get` to a service (REPL command: `param_get`).

Usage:
: `param_get <topic> <key1> <key2> ...`

Requests the specified keys (or entire state if you pass `*`) from
`<topic>/ctl` and queues the reply in the message buffer.

* **Parameters:**
  **arg** – Topic and one or more keys separated by spaces.

#### do_param_set(arg)

Send a control `set` to a service (REPL command: `param_set`).

Usage:
: `param_set <topic> {"key": value, ...}`

Parses the dictionary and sends a `set` control message to
`<topic>/ctl` with the provided key/value pairs.

* **Parameters:**
  **arg** – Topic and dict string.

#### do_quit(arg)

Exit the REPL.

* **Parameters:**
  **arg** – Unused.
* **Returns:**
  True to signal cmd.Cmd to exit.

#### do_send(arg)

Send an arbitrary message (REPL command: `send`).

Usage:
: `send <topic> <msg_type> {"payload": {...}, "meta": {...}}`

Parses a JSON-like dict (via `ast.literal_eval`), extracts `payload` and
`meta`, constructs an `AciesMsg`, and publishes it.

* **Parameters:**
  **arg** – Topic, message kind, and content dict string.

#### do_show(arg)

Wrapper around [`show()`](#acies.core.shell.AciesShell.show) (REPL command: `show`).

Usage:
: `show all` |
  `show topics` |
  `show <topic> [full]`

* **Parameters:**
  **arg** – Target and optional `full` flag.
* **Returns:**
  Optional usage string if the arguments are malformed (printed by cmd.Cmd).

#### do_up(arg)

Enable heartbeat emission (REPL command: `up`).

Sends `{'enable_heartbeat': True}` to `<topic>/ctl`.

* **Parameters:**
  **arg** – Topic string (without or with `/ctl` suffix).

#### emptyline()

Called when an empty line is entered in response to the prompt.

If this method is not overridden, it repeats the last nonempty
command entered.

#### intro *= 'hello!\\n'*

#### postcmd(stop, line)

Hook after each command.

Refreshes the dynamic prompt to include the new-message count.

* **Parameters:**
  * **stop** – The value returned by the command handler.
  * **line** – The raw input line.
* **Returns:**
  The unmodified `stop` value to control REPL exit flow.

#### postloop()

Persist command history and shut down after the REPL exits.

Writes the history file (if `readline` is available) and then calls
[`shutdown()`](#acies.core.shell.AciesShell.shutdown).

#### preloop()

Load command history before entering the REPL loop.

If `readline` is available and a history file exists, load it so that
arrow-key history navigation works.

#### prompt *= 'Acies> '*

#### send(topic, msg_type, payload, meta)

Construct and publish a message via the bound service.

* **Parameters:**
  * **topic** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – Topic to publish to.
  * **msg_type** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – Message kind (e.g., `'json'`, `'get'`, `'set'`,
    `'topic'`, `'reply'`, or `'array_<dtype>'`).
  * **payload** ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict) | [`list`](https://docs.python.org/3/library/stdtypes.html#list) | [`None`](https://docs.python.org/3/library/constants.html#None)) – Message payload (type depends on `msg_type`).
  * **meta** ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict) | [`None`](https://docs.python.org/3/library/constants.html#None)) – Optional metadata to attach.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError) – Propagated if the payload type is invalid for the given kind.

#### show(target, full)

Display messages from the buffer.

* **Parameters:**
  * **target** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – One of `'topics'` (list subscriptions and known topics),
    `'all'` (dump all messages), or a specific topic string.
  * **full** ([`bool`](https://docs.python.org/3/library/functions.html#bool)) – If True, print each message as a full dict; else a compact form.

### Notes

Resets the “new messages” counter after printing relevant messages.

#### shutdown()

Shut down the REPL and its service.

Signals the dispatch thread to stop, calls [`Service.shutdown()`](#acies.core.shell.Service.shutdown), prints
a goodbye message, and sleeps briefly to flush output.

#### update_prompt()

Update the prompt to reflect unseen messages.

Sets the prompt to `'Acies> '` when there are no new messages, or to
`'Acies [ N new msgs ]> '` when the message buffer has grown since the
last [`show()`](#acies.core.shell.AciesShell.show)/print.

### *class* acies.core.shell.Service(conf, \*args, \*\*kwargs)

Bases: [`Service`](acies.core.service.md#acies.core.service.Service)

* **Parameters:**
  **conf** (`Config`)

#### shutdown()

Shutdown the service.

#### start()

Start the service.
