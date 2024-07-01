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



| Scenario | Ingress | Egress| Notes |
|----------|----------|----------|----|
|   Same Region - via Internet Gateway   |   No Cost   |   No Cost   | Data transfer between services within the same AWS region is usually free. |
|   Same Region - via NAT Gataway  |   No Cost   |   See Notes | NAT Gateway Per Hour Service chanrg and per GB processing charge. |
|   Inter Region   |      |   Cost per GB   |Data moves out of a region, you'll be billed for data transfer out|





### Reference

- AWS Blog [link](https://aws.amazon.com/blogs/architecture/overview-of-data-transfer-costs-for-common-architectures/)
