#!/bin/sh

# Process a set of imagery using a one-off cluster

MASTER_INSTANCE_TYPE=m4.large
WORKER_INSTANCE_TYPE=c4.xlarge
WORKER_COUNT=1
# WORKER_COUNT=2
SPOT_WORKER_PRICE=0.15
# SPOT_WORKER_COUNT=1
SPOT_WORKER_COUNT=1

aws s3 cp tiles.txt s3://emr.openterrain.org/gglp/ --acl public-read
aws s3 cp gglp.vrt s3://emr.openterrain.org/gglp/ --acl public-read
aws s3 cp emr.json s3://emr.openterrain.org/gglp/ --acl public-read
aws s3 cp bootstrap.sh s3://emr.openterrain.org/gglp/ --acl public-read
aws s3 cp chunk.py s3://emr.openterrain.org/gglp/ --acl public-read
aws s3 cp pyramid.py s3://emr.openterrain.org/gglp/ --acl public-read

CLUSTER_ID=$(aws emr create-cluster \
  --name "Chunk GGLP" \
  --log-uri s3://emr.openterrain.org/logs/ \
  --release-label emr-4.6.0 \
  --use-default-roles \
  --ec2-attributes KeyName=stamen-keypair,SubnetId=subnet-79698820 \
  --applications Name=Spark \
  --instance-groups \
    Name=Master,InstanceCount=1,InstanceGroupType=MASTER,InstanceType=$MASTER_INSTANCE_TYPE \
    Name=ReservedWorkers,InstanceCount=$WORKER_COUNT,InstanceGroupType=CORE,InstanceType=$WORKER_INSTANCE_TYPE \
    Name=SpotWorkers,InstanceCount=$SPOT_WORKER_COUNT,BidPrice=$SPOT_WORKER_PRICE,InstanceGroupType=TASK,InstanceType=$WORKER_INSTANCE_TYPE \
  --bootstrap-action Path=s3://emr.openterrain.org/gglp/bootstrap.sh \
  --configurations http://s3.amazonaws.com/emr.openterrain.org/gglp/emr.json \
  --auto-terminate \
  --steps \
    Name=CHUNK,ActionOnFailure=TERMINATE_CLUSTER,Type=Spark,Args=[/tmp/chunk.py,/tmp/tiles.txt] \
    Name=PYRAMID,ActionOnFailure=TERMINATE_CLUSTER,Type=Spark,Args=[/tmp/pyramid.py,/tmp/tiles.txt] \
    | jq -r .ClusterId)


echo "Cluster ID: ${CLUSTER_ID}"

# TODO poll cluster status
aws emr describe-cluster --cluster-id $CLUSTER_ID | jq -r '.Cluster.Status.State + ": " + .Cluster.Status.StateChangeReason.Message'
