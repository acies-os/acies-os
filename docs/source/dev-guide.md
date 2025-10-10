# Developer's Guide

```{toctree}
:caption: API Reference
:maxdepth: 2
core-api
```

This section guides you through building and deploying your own service on Acies-OS using the provided API. By subclassing the core Service class and implementing a few required methods, you can integrate custom sensor logic or processing pipelines directly into the middleware.

## 1. Subclass Service

Create a class that inherits from `acies.core.service.Service`.

```python
from acies.core.service import Service

class MyService(Service):
    """Example service that consumes topics and publishes results."""
    ...
```

The base `Service` gives you: a Zenoh session, a control topic (`…/ctl`) with `get/set/topic/reply` handling, a scheduler, pub/sub helpers, and a message queue.

---

## 2. Implement `__init__`

### 2.1 Accept `*args, **kwargs`

Your init **must** accept overflow args to forward into the base `Service` (namespace, proc name, listen/connect, etc.).

### 2.2 Forward to parent

```python
class MyService(Service):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # ← critical: wires up session, ctl topic, queues, etc.
```

### 2.3 Set your publish topic

Use `ns_topic_str(...)` to prepend the configured namespace automatically.

```python
        self.pub_topic = self.ns_topic_str('your_topic')   # e.g., "ns1/vehicle"
        logger.info(f'classification result published to {self.pub_topic}')
```

> Tip: Add defaults to `self.service_states` (thresholds, feature flags) here too.

---

## 3. Implement `handle_message`

Process inbound messages pulled from the internal queue. This is where you parse payloads, apply filters, buffer frames, etc.

You can refer to this example as a blueprint on how to implement your function.

```python
import queue, numpy as np
from datetime import datetime
from acies.core.types import AciesMsg

class Classifier(Service):
    ...
    def handle_message(self):
        """Handle incoming messages from the message queue."""
        try:
            topic, msg = self.msg_q.get_nowait()
            assert isinstance(msg, AciesMsg)
        except queue.Empty:
            return

        # if deactivated, drop the message
        if self.service_states.get('deactivated', False):
            return

        if any(topic.endswith(x) for x in ['geo', 'mic']):
            # msg.timestamp is in ns
            timestamp = int(msg.timestamp / 1e9)
            now = int(datetime.now().timestamp())
            array = np.array(msg.get_payload())
            mod = 'geo' if topic.endswith('geo') else 'mic'

            # filter out low energy messages
            energy = np.std(array)
            thresh = self.service_states.get(f'{mod}_energy_thresh', 0.0)
            if energy < thresh:
                logger.debug(f'energy below threshold: {energy} < {thresh} at {topic}; drop message: {msg}')
                return
            metadata = msg.get_metadata()
            metadata['energy'] = energy

            self.buffer.add(topic, timestamp, array, metadata)
            if self.feature_twin:
                # stage the latest msg for each topic to sync with the twin
                self.twin_sync_register(topic, msg)
        else:
            logger.info(f'unhandled msg received at topic {topic}: {msg}')
```

---

## 4. Implement `run` (schedule your loops)

Use the built-in scheduler to run periodic tasks: message handling, inference, twin sync, etc.

```python
class MyService(Service):
    ...
    def run(self):
        """Main loop: schedule periodic tasks."""
        self.schedule(2, self.log_activate_status, periodic=True)
        self.schedule(0.1, self.handle_message, periodic=True)
        self.schedule(1, self.run_inference, periodic=True)

        if getattr(self, 'feature_twin', False):
            self.schedule(self.sync_interval, self.twin_sync, periodic=True)

        self._scheduler.run()
```

---

## 5. Create a CLI

Use Click plus your project’s `common_options` to expose standard flags (mode, connect, listen, namespace, proc name, etc.).

```python
import click
from acies.core import common_options

@click.command(context_settings=dict(ignore_unknown_options=True))
@common_options
@click.option('--weight', type=str, help='Model weight')
@click.option('--sync-interval', type=int, default=1, help='Twin sync interval (s)')
@click.option('--feature-twin', is_flag=True, default=False, help='Enable digital twin features')
@click.option('--twin-model', type=str, default='multimodal', help='Twin model kind')
@click.option('--twin-buff-len', type=int, default=2, help='Twin buffer length (frames)')
@click.option('--heartbeat-interval-s', type=int, default=5, help='Heartbeat interval (s)')
@click.argument('model_args', nargs=-1, type=click.UNPROCESSED)
def main(mode, connect, listen, topic, namespace, proc_name,
         weight, sync_interval, model_args, feature_twin,
         twin_model, twin_buff_len, heartbeat_interval_s):
    ...
```

Everything from `mode` → `proc_name` is used by the base `Service` to configure the Zenoh session and topic names; the rest are your app’s knobs.

---

## 6. Initialize your class in `main(...)`

Create your service instance and pass all necessary params—**including** the `conf` object (Zenoh config) the base class expects.

```python
from acies.core import get_zconf, init_logger

def main(...):
    # forward extra args to your model if needed
    # update_sys_argv(model_args)  # optional helper if you use it

    init_logger(f'{namespace}_{proc_name}.log', name='acies.infer')
    z_conf = get_zconf(mode, connect, listen)

    svc = MyClassifier(                        # or MyService
        classifier_config_file=weight,
        sync_interval=sync_interval,
        twin_model=twin_model,
        twin_buff_len=twin_buff_len,
        conf=z_conf,                           # ← base Service needs this
        mode=mode,
        connect=connect,
        listen=listen,
        topic=topic,                           # list of topics to pre-subscribe
        namespace=namespace,
        proc_name=proc_name,
        feature_twin=feature_twin,
        heartbeat_interval_s=heartbeat_interval_s,
    )

    # Start the service
    svc.start()
```

---

> Want to customize your service further? Refer to our [API documentation](core-api.md)!
