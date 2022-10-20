# Carbon Aware Scaling Poc
This project is to integrate CO2 emission intensity data into KEDA scaling decision.


# Overview

The background, architecture, findings and demo can be found on the [YouTube playlist](https://www.youtube.com/playlist?list=PLhiQtL6gPrYrdpfhCoxD9ydT6TqNQBA4A)


# Data Sources

- National Grid ESO https://data.nationalgrideso.com/search?q=carbon&sort=score%3Adesc
- Wattime https://www.watttime.org/
- Electricity Maps https://static.electricitymaps.com/api/docs/index.html

## Deployment Scheme
![Deployment](/images/deployment.png)
Relates to https://github.com/kedacore/keda/issues/3467

## Target User Experience:

- Admin can enable Carbon awareness for kubernetes resources, without requiring applications code change.
- Admin defines Carbon Intensity thresholds to scale up/down based on, and AllowedMaxReplicaCount as a scaling target for the Scaled Object (Deployment or StatefulSet).

``` yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: {scaled-object-name}
spec:
  scaleTargetRef:
    name:          {name-of-target-resource}         # Mandatory. Must be in the same namespace as the ScaledObject
  maxReplicaCount:  100                              # Optional. Default: 100
  environmentalImpact:
    carbon:
    - measuredEmission: 300
      allowedMaxReplicaCount: 50
    - measuredEmission: 500
      allowedMaxReplicaCount: 10
  fallback:                                          # Optional. Section to specify fallback options
    failureThreshold: 3                              # Mandatory if fallback section is included
    replicas: 6                                      # Mandatory if fallback section is included
  triggers:
  # {list of triggers to activate scaling of the target resource}
  ```

## Required data 

- **Customer input:** define for the ScaledObject, the desired values for measuredEmission and allowedReplicaCount
- **Platform Metrics:** Location based Carbon Marginal Intensity (also called Electricity Carbon Intensity), expressed in gCO2eq/Kwh (Location based means it depends on the country / Region / Data Center where the K8s cluster is deployed)
 
## Engineering solution

- for POC: start with python
- Frameworks
  - Kubernetes client / sdk
  - Kopf: Kubernetes operator framework --> faciliates writing k8s operators: hooking to k8s resource CRUD + Events and defining handlers for them
    - https://kopf.readthedocs.io/en/stable/timers/
   
Architecture of poc vo

- Operator code: handlers.py + DockerFile: Timer, regularly checks Carbon Intensity and scales up / down, Keda Scaled Object 
- Operator deploy: OperatorDeployment.yaml + rbac.yaml
- Operator target resource: Keda Http scaled object
- Keda scaled Object Target : Dummy Deployment
   
## Open Items

- [ ] Determine initial data provider to integrate
- [ ] Integrate Carbon Intensity Data via K8s Metrics (Prometheus)
- [ ] Update POC to use real-world metrics rather than mock
- [ ] Define proposal for KEDA around Carbon Aware Scaling and how to integrate it as an extension
