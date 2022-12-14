# Getting Started

## Prerequisites

- Kubernetes cluster
- Python3.7 and above
- kopf
- KEDA

## Installation

To prepare the environment for the demo run :

```
deploy/clusterup.sh
```

`clusterup.sh` will deploy `kube-prometheus` operator, `Kepler` and `Carbon-Intensity-Exporter`. The `kube-prometheus` operator in `monitoring` namespace scrapes `Kepler` and `Carbon-Intensity-Exporter` via `ServiceMonitors`.



## Debug

Has my ServiceMonitor been picked up by Prometheus?

ServiceMonitor objects and the namespace where they belong are selected by the serviceMonitorSelector and serviceMonitorNamespaceSelectorof a Prometheus object. The name of a ServiceMonitor is encoded in the Prometheus configuration, so you can simply grep whether it is present there. The configuration generated by the Prometheus Operator is stored in a Kubernetes Secret, named after the Prometheus object name prefixed with prometheus- and is located in the same namespace as the Prometheus object. For example for a Prometheus object called k8s one can find out if the ServiceMonitor named `kepler-exporter` has been picked up with:

```
kubectl -n monitoring get secret prometheus-k8s -ojson | jq -r '.data["prometheus.yaml.gz"]' | base64 -d | gunzip | grep "kepler-exporter"

```