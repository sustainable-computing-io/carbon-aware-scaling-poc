KEDA + Sustainability Meeting Minute
===

###### tags: `sustainability` `KEDA`

:::info
- **Date:** Sept 1, 2022 8:30 PM (ET)
- **Agenda**
1. CO2 Emission Data Provider `10min`
2. Pod Power Measurement Metrics `10min`
3. VPA/VFA for KEDA `10min`
- **Participants:**
    - Red Hat: 
        - Huamin Chen
        - Zbynek Roubalik
        - William Caban
    - IBM
        - Chen Wang
    - Microsoft
        - Tom Kerkhove
- **Contact:** 
:::

## CO2 Emission Data Provider
- Support multiple CO2 emission APIs (co2signal, watt time)
- Need a shared repo to include CO2 emission and location, consistent schema, multiple data sinks (REST, Kafka, Promethus)
- Could collaborate under CNCF Sustainability TAG
    - Data source, standards, API scheme spec
    - OpenTelemetry support
    - WeaveWorks experience: these APIs need to incorporate location
    - KEDA: native (REST) extension for target spec to obtain the metrics
    - Tom/Zbynek to explore an early PoC
    - an prototype of using CO2signal is at https://github.com/redhat-et/kepler-os-climate-integration/blob/main/kafka/producer/co2signal_kafka_producer.py
## Pod Power Measurement Metrics
- Measuring power consumed by Pods, processes, containers
    - including CPU/GPU, RAM, etc
- Methodology validated by scientific research, support both baremetal and VM
- Consistent power consumption metrics schema, support multiple data sinks
- KEDA uses default HPA for scheduling
    - Deployment can use custom scheduler and the deployments are managed by KEDA
    - Use case of annotation to the deployment needs justification
    - Scheduler vs scaling question: resource quota are managed by schedulers, scaling is managed by KEDA.

## VPA for KEDA
- KEDA doesn't support VPA yet. Consider tuning resources for Pods for sustainability purpose
    - KEDA can scale up and down based on e.g. Kepler Prometheus metrics
    - KEDA scale in/out if CO2 consumpion is up/down (based on end user formula)
    - VPA in backlog, no active plan, but maybe VPA can manage deployment separately
- Likewise, frequency tuning can also be a sustainability option


:::info
- **Date:** Sept 19, 2022 8:30 PM (ET)
- **Agenda**
1. CO2 Emission Data Provider `10min`
2. Pod Power Measurement Metrics `10min`
3. VPA/VFA for KEDA `10min`
- **Participants:**
    - Red Hat: 
        - Huamin Chen
        - Zbynek Roubalik
    - IBM
        - Chen Wang
    - Microsoft
        - Tom Kerkhove
        - Yassine El Ghali
        - Vaughan Knight
- **Contact:** 
:::

## Items
- KEDA focus on scalability
- Need power measurement (with and without co2)
- CO2 emission API/SDK: current data and forecast data, time shift (i.e. best time for cron jobs)
- KEDA focused items:
    - actions to reduce co2, how to measure saving: scaling and reporting
    - KEDA emits co2 event to trigger scale up/down, other components provide reporting
        - event contains duration and co2 intensity. It is defined as CRD (e.g. 80% co2 threshold), https://github.com/kedacore/keda/issues/3467
        - PoC requirement: needs co2 data API (either from customer or from the provider). The data source are not with the keda core. PoC will discover the data source/api gaps and provide evidence for future investigation 
        - Starting from ElectricityMap API, exporting to Prometheus, KEDA queries Prometheus 
            - https://github.com/Green-Software-Foundation/carbon-aware-sdk
        - focused on the frontend API (i.e. prometheus query), backend source and framework can be pluggable.
        - Zbynek will sync up with Huamin on development, Yassine will join. PoC by KubeCon, to demo at (TBD) vendor booths
            - https://github.com/sustainable-computing-io/carbon-aware-scaling-poc/
- VPA, VFA (CPU frequency scaling)


