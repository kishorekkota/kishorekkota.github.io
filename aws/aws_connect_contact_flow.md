## Amazon Connect Contact Flow

A contact flow defined customer interaction with contact center, such as playing prompts, collecting user input, and transferring calls.

It define routing logic for customer interactions, such which agent, queue the call routes. These are decided based on customer data, input and priority.

Contact flow can incorporate decision based logic, which is if/then conditions. Checking for input, checking for time and agent availability.



### Inbound Contact Flow / Generic Contact Flow

Handles Inbound Customer Interactions

This is the entry point Inbound call, we will start with an entry point, depending on the requirement we can define `set logging and analytics behavior`, then we flow to a Prompt for greeting customer.

User input can be collected via `Get Customer Input` block, this allows for collecting input and play respective message.

Based on option selected, we can define `set working queue` block. Then check for `Hours of Operation`, then we need to proceeded to `Transfer to Queue` block.



### Customer Queue Flow

Customer Queue flow 

Default Customer Queue - This get invoked when a call is transferred to Queue while waiting for Agent to be connected.

Set Customer Queue Flow can be set prior to Transfer Queue, as long as it is executed prior to Transfer Queue flow, this will get executed.

Default Customer Queue flow simply contains Entry Point and Loop Prompt, which plays hold music. This invokes automatically while the customer is waiting to be connected to agent.




### Different Types of Contact Attributes

Contact Attribute is a key value pair that contains data about a contact.

System
Agent
Queue Metrics
Customer - will be available when customer profile block is used.   
Media Stream
Lex Slots
Lex Attributes
External
User-Define