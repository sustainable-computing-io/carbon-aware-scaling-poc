import os
import kopf
import kubernetes
import yaml
import asyncio
import time
import requests


def getCarbonIntensity():
    response = requests.get('https://greenapimockyaya.azurewebsites.net/api/CarbonRating')
    json_data = response.json() if response and response.status_code == 200 else None
    carbon_rating = json_data['Rating'] if json_data and 'Rating' in json_data else None
    return carbon_rating 


@kopf.on.event('deployment',
                labels={'carbon-aware': 'yes'})
def my_handler(event, **_):
    print(event)


#Carbon Aware scaling
@kopf.timer('scaledobject', interval=5.0, 
                labels={'carbon-aware': 'yess'})
def carbonAwareKeda(body, spec, name, namespace, status, **kwargs):

    carbon_rating = getCarbonIntensity()
    if carbon_rating > 500:
        maxReplicaTarget = 3
        eventReason = "CarbonBasedScaleDown"
        eventMessage = "Scaled Down Deployment to 3"
    else:
        maxReplicaTarget = 10
        eventReason = "CarbonBasedScaleUp"
        eventMessage = "Scaled Up Deployment to 10"


    keda_scaledobject_patch = {'spec': {'maxReplicaCount': maxReplicaTarget}}

    api = kubernetes.client.CustomObjectsApi()
    obj = api.patch_namespaced_custom_object(
        group="keda.sh",
        version="v1alpha1",
        name="http-scaledobject",
        namespace="default",
        plural="scaledobjects",
        body=keda_scaledobject_patch,
    )     

    kopf.info(body, reason=eventReason, message=eventMessage)

