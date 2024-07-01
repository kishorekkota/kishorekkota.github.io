---
title: AWS Data xFer Cost
layout: home
nav_order: 10
---

Desribes cost of data transfer when using AWS.

### Scenarios

- **Same Region** - Cost of data transfer between services deployed in the same region.
- **Inter Region** - Cost of data transfer between services deployed across two different region.
- **Internet** - Cost of data transfer between AWS Services and outside services when connecting via Internet.
- **AWS Direct Connect** - Cost of data transfer between AWS Service and enterprise data center when leveraging AWS DC.



### Cost

Highlevel Data Transfer Cost


| Scenario | Ingress | Egress| Notes |
|----------|----------|----------|----|
|   Same Region - via Internet Gateway   |   No Cost   |   No Cost   | Data transfer between services within the same AWS region is usually free. |
|   Same Region - via NAT Gataway  |   No Cost   |   See Notes | NAT Gateway Per Hour Service chanrg and per GB processing charge. |
|   Inter Region   |   No Cost   |   Cost per GB   |Data moves out of a region, you'll be billed for data transfer out|
|  Internet   |   No Cost   |   Cost per GB   | Data In via Internet has no charge, Data Out will be charged per GB. |


EC2 Data Transfer Cost 

- EC2 Data Transfer Cost [link](https://aws.amazon.com/ec2/pricing/on-demand/#Data_Transfer/)

| Scenario | Ingress | Egress| Notes |
|----------|----------|----------|----|
|Internet| No Cost | Cost per GB | Data Out is from EC2 is charged |
|Inter Region | No Cost | Cost per GB | Data Out is charged |
|Same Region | Cost Per GB | Cost per GB | Data In and Data Out are charged. $0.01 per GB both directions. |
| Same AZ | No Cost | No Cost | Data transfer within the same AZ is free. |
| EC2 to S3/EBS/Dynamo DB/SES/SQS/Kinesus/ECR/SNS| No Cost | No Cost | If other AWS services are in the path of your data transfer, you will be charged their associated data processing costs. These services include, but are not limited to, PrivateLink endpoints, NAT Gateway and Transit Gateway. |




### Reference

- AWS Blog [link](https://aws.amazon.com/blogs/architecture/overview-of-data-transfer-costs-for-common-architectures/)
