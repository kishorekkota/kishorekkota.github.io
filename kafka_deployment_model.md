---
layout: default
title: Kafka Deployment Model
---


### Kafka in Enterprise Environments: Shared vs. Dedicated Infrastructure Models

## Introduction
This document aims to provide a comprehensive comparison of using Apache Kafka in a shared infrastructure model versus a dedicated infrastructure model for each team within an enterprise.

This document strictly limited to Kafka Tenancy Model and Comparision between Multi Tenancy and Dedicated Tenancy model.


## Out of Scope

- Does not cover details around Kafka usage or deployment architecture.

### Kafka and Multitenancy
Apache Kafka, while not inherently designed for multitenancy, possesses features that make it suitable for such a setup within an enterprise. Multitenancy in Kafka involves a single Kafka cluster being used by different systems or teams.

#### Supporting Features for Multitenancy
1. **Topic Segregation**: Organizes messages into topics for different teams or applications.
2. **Access Control Lists (ACLs)**: Controls access to specific topics.
3. **Quotas**: Sets limits on resource usage for clients or users.
4. **Scalability**: Kafka's distributed nature allows handling increased loads.
5. **Performance Isolation**: Achieved through careful planning and resource allocation.
6. **Monitoring and Logging**: Essential for tracking usage and diagnosing issues.

#### Challenges in Multitenancy
- **Resource Contention**: Risk of one tenant’s usage impacting others.
- **Maintenance and Upgrades**: Complexity in managing without service disruption.
- **Security and Compliance**: Increased challenge in a shared environment.

---

## Kafka as a Shared vs. Dedicated Infrastructure Model

### Shared Infrastructure Model
#### Pros
- Cost Efficiency 
  * Overall capacity needed can be optimized and efficient use of the resources thus providing cost efficiency.
- Easier Maintenance
  * Allows for dedicated enterprise team to manage Kafka Infrastructure. This takes away burden from each and every team trying to manage Kafka on their own.
- Resource Optimization
  * Shared infrastruture has an advantage to limit underutilization of resources compared to having seperate instaces.

#### Cons
- Risk of Resource Contention
  * As the resource limits are not supported by Kakfa deployment model, this is a possible scenario - a specific teams deployment may cause resource exchaustion due to incorect planning or unexpected scenario.  
- Limited Customization
  * Shared infra limits all teams to be on the same Kafka version and limits any customization, or some use cases might be requiring higher or lower replication factor or availability needs being different, etc.
- Potential Security Concerns
  * Does not enforce RBAC at the topic level, this can lead to unauthorized consumption of a topic. Also, these will end up adding as requirement for infrastructure to enforce prior autherization of events as part of consumer onboarding.

### Dedicated Infrastructure Model
#### Pros
- Customization
 * Teams have authroity to defined Kafka infrastructure based on the specific availability and realiability needs.
- Enhanced Security
 * Establishes isolation of Kafka between teams thus limiting unautherized access exposure. 
- Performance Reliability
 * Dedicated resource ensure teams are have dedicated capacity for running the workloads.

#### Cons
- Higher Costs
- Increased Maintenance Effort
- Underutilization of Resources

### Comparative Analysis
Comparison of shared vs. dedicated models in terms of cost, performance, security, scalability, and maintenance.



| Criteria         | Shared Tenancy | Dedicated Tenancy | Notes |
|------------------|---------------------|----------------------|-----------------|
| Cost             | [x]             | ...                  | ...             |
| Performance  | ...                 | [x]                 | ...             |
| Reliability              | ...                 | [x]                | ...             |




### Recommendations
Suggestions based on different enterprise sizes and needs, with decision factors for consideration.

### Conclusion
Summary of key findings and final thoughts.
