# AciesOS V2 Design Document

## API

The version 2 of `acies-os` provides a decorator pattern to the library user,
similar to Flask and FastAPI:

```python
from acies import Acies
app = Acies()
```

The communication system is based on a publish-subscribe model, where the user
can subscribe to specific topics and receive messages when they are published.
The user can also publish messages to specific topics.

There are three types of handlers that can be defined by the user:

```python
# Functions that handle messages from specific topics
@app.subscribe(topics=["node1/geo", "node1/mic"])
def handle_mic_and_geo_data(msg, ctx: AciesContext):
    print(f"Received seismic msg: {msg}")

# Functions that run periodically, regardless of incoming messages
@app.schedule(interval=5)
def try_run_inference(ctx: AciesContext):
    print("Running inference every 5 seconds")

# Functions that produce messages, for example from a sensor or an external API
@app.produce()
```

The `app.on()` decorator takes the following parameters:

```python
@app.on(
    topics: list[str],  # List of topics to subscribe to
    priority: int = 0,  # Priority of the handler, higher priority handlers will be called first
    timeout: float | None = None,  # Optional timeout for the handler, in seconds
    max_wait: float | None = None,  # Optional maximum wait time for the handler, in seconds
)
```

## Internal Design

```python
@dataclass
class TaskSpec:
    name = str | None = None
    kind: str = 'subscribe'  # 'subscribe' | 'schedule' | 'produce'
    topics: list[str] = field(default_factory=list)
    schedule: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    fn: Callable[..., Any] = lambda: None

    def __call__(self, fn):
        self.fn = fn
        return fn

    def make_job(self, *, payload=none, now=None, key=None):
        return Job(task=self, palyload=payload, key=key, created_at=now)


@dataclass
class Job:
    task: Task
    payload: Any
    key: Any | None = None
    created_at: float = field(default_factory=time.time)


class AciesContext:
    def emit(self, topic: str, msg: Any):
        """In-process publish-subscribe mechanism, allows handlers to emit
        messages to other handlers."""
        pass

    def publish(self, topic: str, msg: Any):
        """Network publish-subscribe mechanism, allows handlers to publish
        messages to other nodes."""
        pass


class Acies:
    def __init__(self, max_workers: int = 10):
        self.max_workers: int = max_workers
        self.routes: dict[str, list[TaskSpec]] = field(default_factory=dict)
        self.executor: ThreadPoolExecutor = field(
            default_factory=lambda: ThreadPoolExecutor(max_workers=max_workers))
        self.jobs: list[Job] = field(default_factory=list)
        self.ingress_queue: Queue = field(default_factory=Queue)

    def subscribe(
            self,
            topics: list[str],
            *,
            trigger: str | None = None,
            priority: int = 0,
            max_wait: float | None = None,
            timeout: float | None = None):
        pass

    def schedule(
            self,
            *,
            interval: float | None = None,
            delay: float | None = None,
            priority: int = 0,
            timeout: float | None = None):
        pass

    def produce(
            self,
            *,
            name: str | None = None,
            restart: bool = True,
            backpressue: str = 'block',
            queue_size: int | None = None,
            daemon: bool = True):
        pass

    def run(self):
        pass
```
