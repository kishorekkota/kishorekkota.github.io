---
title: Compensating Transactions
layout: home
parent: Microservice
---
## Compensating Transactions

Compensating transactions are a pattern used in distributed systems to handle the rollback of a previously completed transaction. They are typically used in scenarios where a transaction involves multiple steps or services, and if any of these steps fail, the entire transaction needs to be rolled back.

The idea behind compensating transactions is to have a separate set of actions that can undo the effects of a completed transaction. These compensating actions are designed to reverse the changes made by the original transaction, ensuring that the system is left in a consistent state.

Compensating transactions are often implemented using a two-phase commit protocol. In the first phase, all participants in the transaction prepare for the commit by ensuring that they can successfully complete their part of the transaction. If any participant fails to prepare, the transaction is aborted. In the second phase, if all participants have successfully prepared, they commit the transaction. If any participant fails to commit, they execute their compensating actions to undo the changes made by the transaction.

Using compensating transactions can help ensure data consistency and integrity in distributed systems. They provide a way to handle failures and rollbacks in a controlled manner, allowing the system to recover from errors and maintain a consistent state.
