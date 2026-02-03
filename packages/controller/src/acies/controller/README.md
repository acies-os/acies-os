# Fault Tolerance
Here we provide the overall design of the fault tolerance and further provide the detailed steps to conduct failover experiments.

The key components of the fault tolerance are: 

1. Monitoring: the controller collects system metrics (including heartbeats) of the worker nodes by subscribing topics **
2. Anomaly Detection: the controller detects the failures by checking the states of nodes and services. Both the nodes and services need to be registered and recorded in file `config/execution_plan.json`. In current version, three types of failures are supported (described in [Supported Failures](#supported-failures)).
3. Failover: the controller mitigates the failures using different failover strategies, including failover to a backup service/model/node. For each type of failure, the failover strategy is described in [Supported Failures](#supported-failures). 

To enable the fault tolerance, we need to manually configure the execution plan and the backup plan in the folder `config`. (Note that, we have plan to automate the configuration and make our controller adaptive to the system dynamics)


## Supported Failures
We support four failure types. Here we describe the failures, the way to detect them, and the way to recover (failover).
1. node failure (`node`): a failure of node, including both sensors and services. We detect it by comparing the node list reported by monitoring tools with the registered nodes defined in `config/execution_plan.json`. When a node failure is detected, our controller activate a backup node by looking up the backup plan in `config/backup_plan.json`
2. service failure (`infer`): a failure of infer service. We detect it by comparing the infer services reported by monitoring tools with the registered infer services defined in the execution plan. We a infer service failure is detected, our controller activate its backup service in another node, meanwhile, update the bakcup service to subscribe the topics of sensors (mic and geo). The bakcup service is defined in our backup plan.
3. sensor failure (`mic` or `geo`): a failure of sensor. We detect the sensor failure by detecting the failure of its service (named `mic` and `geo`). The detection approach is the same as the way to detect service failure. Once a sensor failure is detected, the controller activate a backup infer service on other node, which is a uni-model inference service. The backup uni-model inference service is defined in our backup plan. 


## How To 
To demonstrate the fault tolerance of our controller, the steps are below: 
1. Make execution and backup plan. You need to design the topology of sensors, services, and nodes in the network.
2. Configure the execution and backup plan. The examples are provided in `config/examples`
3. Deploy your services and nodes as designed (including the backup services and nodes)
4. Trigger a failure described in [Supported Failures](#supported-failures) by fault injection methods like unplug a node, kill a service, or unplug a sensor. 
5. The controller detects the failure and triggers the failover
6. Monitor the system states after failover
7. If you want to conduct another failover experiment, go to step 2 with the new system states you monitored in step 6. If you have a new execution or backup plan, then restart from step 1. 
