#!/bin/sh

# Process a set of imagery using a one-off cluster

MASTER_INSTANCE_TYPE=m3.xlarge
WORKER_INSTANCE_TYPE=c3.xlarge
WORKER_COUNT=1
# WORKER_COUNT=2
SPOT_WORKER_PRICE=0.15
# SPOT_WORKER_COUNT=1
SPOT_WORKER_COUNT=5

CLUSTER_ID=$(aws emr create-cluster \
  --name "Convert NED img to TIFF" \
  --log-uri s3://emr.openterrain.org/logs/ \
  --release-label emr-4.2.0 \
  --use-default-roles \
  --auto-terminate \
  --ec2-attributes KeyName=stamen-keypair \
  --instance-groups \
    Name=Master,InstanceCount=1,InstanceGroupType=MASTER,InstanceType=$MASTER_INSTANCE_TYPE \
    Name=ReservedWorkers,InstanceCount=$WORKER_COUNT,InstanceGroupType=CORE,InstanceType=$WORKER_INSTANCE_TYPE \
    Name=SpotWorkers,InstanceCount=$SPOT_WORKER_COUNT,BidPrice=$SPOT_WORKER_PRICE,InstanceGroupType=TASK,InstanceType=$WORKER_INSTANCE_TYPE \
  --bootstrap-action Path=s3://emr.openterrain.org/ned/bootstrap.sh \
  --configurations http://s3.amazonaws.com/emr.openterrain.org/ned/emr.json \
  --steps \
    Name="Reproject",ActionOnFailure=TERMINATE_CLUSTER,Type=STREAMING,Args=[-files,s3://emr.openterrain.org/ned/reproject-ned.sh,-mapper,reproject-ned.sh,-input,s3://emr.openterrain.org/ned/input.txt,-output,s3://emr.openterrain.org/ned/output] \
    | jq -r .ClusterId)

echo "Cluster ID: ${CLUSTER_ID}"

# TODO poll cluster status
aws emr describe-cluster --cluster-id $CLUSTER_ID | jq -r '.Cluster.Status.State + ": " + .Cluster.Status.StateChangeReason.Message'

# TODO fetch $WORKSPACE_URI/step1_result.json
