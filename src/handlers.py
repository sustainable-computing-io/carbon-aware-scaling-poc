import os
import kopf
import kubernetes
import yaml
import asyncio
import time
import requests
import json

prom_endpoint = os.getenv("PROM_ENDPOINT")
prom_query = os.getenv("PROM_QUERY", default="carbon_intensity")

def getCarbonIntensityFromProm():
    params={
        'query': prom_query
    }
    response = requests.get(prom_endpoint+'/api/v1/query', params=params)
    results = response.json()['data']['result']
    if len(results) > 0:
        carbon_intensity_value = results[0]['value'][1]
        return carbon_intensity_value
    return None

def getCarbonIntensityFromCarbonRating():
    response = requests.get('https://greenapimockyaya.azurewebsites.net/api/CarbonRating')
    json_data = response.json() if response and response.status_code == 200 else None
    carbon_rating = json_data['Rating'] if json_data and 'Rating' in json_data else None
    return carbon_rating 

def getCarbonIntensity():
    return getCarbonIntensityFromCarbonRating()
##############################


def get_existing_scaledObject(scaledObjectsList, name):     
    for scaledObject in scaledObjectsList["items"]:
        scaledObjectName = scaledObject['metadata']['name']
        if scaledObjectName == name:
            return scaledObject
    return None


def patch_namespaced_scaledObject(api, namespace, scaledObject, patch_data):
    obj = api.patch_namespaced_custom_object(
        group="keda.sh",
        version="v1alpha1",
        name="httpj",
        namespace="default",
        plural="scaledobjects",
        body=patch_data,
    )     
    return obj

def list_scaledObjects_in_namespace(api, namespace):
    scaledObjects = api.list_namespaced_custom_object(
        group="keda.sh",
        version="v1alpha1",
        namespace="default",
        plural="scaledobjects",
    )     
    return scaledObjects

def list_carbonAwareScalers_in_namespace(api, namespace):
    #carbonAwareScalersList = api.list_namespaced_custom_object(
    carbonAwareScalersList = api.list_cluster_custom_object(
        group="carbon-aware-actions.cncf",
        version="dev",
        #namespace="default",
        plural="carbonawarescalers",
    )     
    return carbonAwareScalersList

def get_existing_carbonAwareScaler(carbonAwareScalersList, name):     
    for carbonawarescaler in carbonAwareScalersList["items"]:
        targetScaledObjectName = carbonawarescaler['spec']['kedaScaledObjectRef']['name']
        if targetScaledObjectName == name:
            return carbonawarescaler
    return None


##############################

@kopf.index('carbonawarescalers.carbon-aware-actions.cncf')
def carbonawarescalers_idx(namespace, name, spec, **kwargs):
    
    scalingRules = spec['kedaScaledObjectsRef']['scalingRules']
    sortedScalingRules = sorted(scalingRules, key=lambda k: k['carbonIntensity']) 

    carbonIntensityValues = [ rule["carbonIntensity"] for rule in sortedScalingRules]
    allowedMaxReplicaCountValues = [ rule["allowedMaxReplicaCount"] for rule in sortedScalingRules]

    return {(namespace, name) : {"carbonIntensityInputValues": carbonIntensityValues, "allowedMaxReplicaCountInputValues": allowedMaxReplicaCountValues} }



@kopf.index('scaledobjects', 
                annotations={'carbon-aware-actions.actionkind': 'carbonawarescaler'})
def scaledobjects_idx(namespace, name, annotations, **kwargs):
    
    carbonawarescalerName = annotations["carbon-aware-actions.actionRef"]
    return {(namespace, name) : {"carbonawarescaler": carbonawarescalerName } }


##############################

@kopf.on.startup()
def configure(settings: kopf.OperatorSettings, **_):
    if os.environ.get('ENVIRONMENT') is None:
        # Only as an example:
        settings.admission.server = kopf.WebhookAutoTunnel()
        settings.admission.managed = 'auto.kopf.dev'
    else:
        # Assuming that the configuration is done manually:
        settings.admission.server = kopf.WebhookServer(addr='0.0.0.0', port=8080)
        settings.admission.managed = 'auto.kopf.dev'


##############################

@kopf.on.mutate('scaledobjects')
def annotate_current_scaledobject(body, old, new, spec, name, namespace, annotations, patch, operation, **kwargs):

    if operation == "CREATE":

        api = kubernetes.client.CustomObjectsApi()

        # list the CarbonAwareScalers within the same namespace
        carbonAwareScalersList = list_carbonAwareScalers_in_namespace(api, namespace)

        targetCarbonAwareScaler = get_existing_carbonAwareScaler(carbonAwareScalersList, name)     
        if targetCarbonAwareScaler is None : return

        annotations_dict = {
                    'carbon-aware-actions.actionkind': "carbonawarescaler",
                    'carbon-aware-actions.actionRef': targetCarbonAwareScaler['metadata']['name'],
                    'carbon-aware-actions.param.defaultMaxReplicaCount': str(spec['maxReplicaCount'])
                    }
        patch.metadata['annotations'] = annotations_dict

##############################

@kopf.on.create('carbonawarescaler.carbon-aware-actions.cncf')  
def annotate_target_scaledobjects(body, spec, name, namespace, **kwargs):

    api = kubernetes.client.CustomObjectsApi()

    annotations_dict = {
                'carbon-aware-actions.actionkind': 'carbonawarescaler',
                'carbon-aware-actions.actionRef': name
            }

    #get target Keda ScaledObject for current carbonAwareScaler
    targetScaledObjectName = spec['kedaScaledObjectRef']['name']

    # list the scaledObjects within the same namespace
    scaledObjectsList = list_scaledObjects_in_namespace(api, namespace)

    targetScaledObject = get_existing_scaledObject(scaledObjectsList, targetScaledObjectName)     
    if targetScaledObject is None : return

    # add originalMaxReplicaCount
    annotations_dict.update({
                'carbon-aware-actions.param.defaultMaxReplicaCount': str(targetScaledObject['spec']['maxReplicaCount']),
            })
    annotations = {'metadata': {'annotations': annotations_dict } }
    obj = patch_namespaced_scaledObject(api, namespace, targetScaledObject, annotations)

##############################

@kopf.on.delete('carbonawarescaler.carbon-aware-actions.cncf')  
def restore_scaledObject_config(body, spec, name, namespace, **kwargs):
    api = kubernetes.client.CustomObjectsApi()

    # remove annotations
    annotations_dict = {
                'carbon-aware-actions.actionkind': None,
                'carbon-aware-actions.actionRef': None,
                'carbon-aware-actions.param.defaultMaxReplicaCount': None,
            }


    #get target Keda ScaledObject for current carbonAwareScaler
    targetScaledObjectName = spec['kedaScaledObjectRef']['name']

    # list the scaledObjects within the same namespace
    scaledObjectsList = list_scaledObjects_in_namespace(api, namespace)

    targetScaledObject = get_existing_scaledObject(scaledObjectsList, targetScaledObjectName)     
    if targetScaledObject is None : return


    # restore originalMaxreplicaCount
    defaultMaxReplicaCount = targetScaledObject['metadata']['annotations'].get("carbon-aware-actions.param.defaultMaxReplicaCount", None) 
    if defaultMaxReplicaCount is None : return

    # remove annotations + restore originalMaxReplicaCount
    scaledObject_patch = {'spec': {'maxReplicaCount': int(defaultMaxReplicaCount) }, 
                          'metadata': {'annotations': annotations_dict} 
                         }

    obj = patch_namespaced_scaledObject(api, namespace, targetScaledObject, scaledObject_patch)

##############################

"""
@kopf.on.event('scaledobjects')
def my_handler(event, **_):
   
    print(event)
"""

##############################

#Carbon Aware scaling
@kopf.timer('scaledobject', interval=10.0, 
                annotations={'carbon-aware-actions.actionkind': 'carbonawarescaler'})
def carbonAwareKeda(carbonawarescalers_idx: kopf.Index, body, spec, name, namespace, status, annotations, **kwargs):

    carbonawarescalerName = annotations.get("carbon-aware-actions.actionRef", None) 
    defaultMaxReplicaCount = annotations.get("carbon-aware-actions.param.defaultMaxReplicaCount", None) 

    # TODO: change to namespaced CRD: scalingRules = carbonawarescalers_idx.get((namespace, carbonawarescalerName))
    scalingRules = carbonawarescalers_idx.get((None, carbonawarescalerName))
    if scalingRules is None: return
    rules = [r for r in scalingRules][0]   #ScalingRules is an iterator, cannot be accessed directly

    carbonIntensityValues = rules["carbonIntensityInputValues"]
    allowedMaxReplicaValues = rules["allowedMaxReplicaCountInputValues"]
    
    carbon_rating = getCarbonIntensity()

    if carbon_rating < min(carbonIntensityValues): #use orginalMaxReplicaCount as it is the highest value
        maxReplicaTarget = int(defaultMaxReplicaCount)

    elif carbon_rating >= max(carbonIntensityValues): #use smallest value
        maxReplicaTarget = allowedMaxReplicaValues[-1]

    else: #Use closest value for current carbonIntensity
        i=0
        while (carbon_rating > carbonIntensityValues[i]) : i=i+1
        maxReplicaTarget = allowedMaxReplicaValues[i-1] # list starts with O

    keda_scaledobject_patch = {'spec': {'maxReplicaCount': maxReplicaTarget}}

    api = kubernetes.client.CustomObjectsApi()
    obj = patch_namespaced_scaledObject(api, namespace, name, keda_scaledobject_patch)

    eventReason = "CarbonBasedScaleDown"
    eventMessage = "Scaled Down Deployment to 3"
    kopf.info(body, reason=eventReason, message=eventMessage)

