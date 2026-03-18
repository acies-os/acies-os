# Acies-OS 2.0 Design Document

## Overview

Acies-OS is a middileware designed for Edge AI applications. It assumes a
computation graph model where nodes represent computational tasks and edges
represent data dependencies between these tasks. The communication layer is
based on a pub-sub system.

The refactor of the 2.0 version is to:

1. Upgrade the pubsub layer to the latest version (zenoh >= 1.0)
2. Provide a decorator-based API for users to define their computation graph
   (akin to Ray, Flask, FastAPI, etc.)
3. Provide internal abstractions for future research and improvements, such as
   different scheduling policies.

The user code creates an instance of `AciesApp`, then defines functions that
represent computational tasks, decorated by decorators provided by the library.
We support 4 types of decorators:

- Type 1 (source nodes): functions that produce messages, for example from a
  sensor or an external API
- Type 2 (sink nodes): functions that handle messages from specific topics, but
  do not produce any messages
- Type 3 (transform nodes): functions that handle messages from specific topics
  and produce messages to other topics
- Type 4 (service / rpc nodes): functions that can be called by other nodes,
  and may return a response

Type 2 and Type 3 nodes should support both event-driven execution (triggered
by incoming messages) and periodic execution (triggered by a timer). For
periodic execution, the function will be called at a specified interval,
regardless of incoming messages. Type 4 can be implemneted using zenoh's
query/queryable mechanism.

In addition to the four task types, the framework supports `on_startup` and
`on_shutdown` lifecycle hooks, which are executed when the application starts
and shuts down, respectively. These functions can be used for initialization
and cleanup tasks.

The library exposes its functionality through a `AciesContext` object that is
passed to the user-defined functions. This context provides methods for
publishing messages, subscribing to topics, and accessing other functionalities
of the library.

The decorated functions produce `TaskSpec` and are registered in a task
registry, which is used by the library to manage the execution of the tasks.
Whenever a message is received on a topic or a timer is triggered, a `Job` is
created based on the corresponding `TaskSpec` and scheduled for execution. The
queueing and scheduling of jobs are managed by a scheduler, which supports
different scheduling policies (e.g., FIFO, priority-based, etc.). The execution
of jobs is handled by an executor, which may use a thread pool or other
concurrency mechanisms to execute tasks in parallel.

Internally, `AciesApp` contains

- A Messaging Layer that serves as an indrection layer to the underlying
  transport layer (zenoh for pubsub, maybe IPC and other protocols), as well as
  handle ingress and egress of messages to allow optimization and better
  mocking for testing.
- A Scheduler that owns ready queues for jobs that are ready to be executed,
  handling the scheduling policies, prioritization and dispatch decisions.
- An Executor that owns worker threads (including thread pool for job execution
  and message threads for handling incoming messages) and dispatch/execute jobs
  based on the scheduling decisions from the scheduler.
- A set of TaskSpecs that handles control plane messages such as change of
  subscriptions, timers, get and set node states.
