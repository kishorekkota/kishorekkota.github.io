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



---------

# Amazon Connect Contact Flow: Step-by-Step Guide

Creating a **Contact Flow** in Amazon Connect involves connecting various blocks that manage interactions with customers. Below is a step-by-step guide on structuring a basic Contact Flow with key blocks in order.

---

## 1. Entry Point
- **Block**: **Start Block**
  - **Description**: The initial block where the Contact Flow begins. No specific settings are needed here.
  
---

## 2. Greet or Provide Information
- **Block**: **Play Prompt** (under "Interact")
  - **Description**: Plays a recorded message or text-to-speech prompt to greet the customer.
  - **Settings**: 
    - Add an audio file or enter text for the text-to-speech conversion.
  
---

## 3. Collect Customer Input (Optional)
- **Block**: **Get Customer Input** (under "Interact")
  - **Description**: Collects customer responses via keypad (DTMF) or spoken input, typically for IVR menus.
  - **Settings**: 
    - Configure response type (DTMF or Speech) and set a timeout duration.

---

## 4. Branching Logic for Customer Input
- **Block**: **Check Contact Attributes** (under "Check")
  - **Description**: Routes the customer based on input or attributes (e.g., “Press 1 for Sales”).
  - **Settings**: 
    - Define conditions like “If input = 1” or “If input = 2.”

---

## 5. Route Call Based on Input
- **Block**: **Transfer to Queue** (under "Integrate")
  - **Description**: Sends the customer to a specific queue based on their input (e.g., Sales or Support).
  - **Settings**: 
    - Select the appropriate queue and routing profile.

---

## 6. Queue Management
- **Block**: **Set Customer Queue Flow** (Optional)
  - **Description**: Applies custom queue management flows (e.g., play music, provide wait times).
  - **Settings**: 
    - Choose an existing queue flow or create a new one.

---

## 7. Check Agent Availability
- **Block**: **Check Queue Metrics** (under "Analyze")
  - **Description**: Checks real-time queue metrics, like agent availability, to make routing decisions.
  - **Settings**: 
    - Select the metric to monitor (e.g., available agents).

---

## 8. Logic Block for Call Routing
- **Block**: **Branch** (under "Logic")
  - **Description**: Creates decision points based on conditions like queue status or customer attributes.
  - **Settings**: 
    - Define logic rules (e.g., “If available agents < 1” or “If customer is VIP”).

---

## 9. Failover or Fallback Options
- **Block**: **Transfer to Flow** (under "Integrate")
  - **Description**: Transfers the customer to another flow, such as an escalation flow.
  - **Settings**: 
    - Specify the target Contact Flow.

---

## 10. End the Call
- **Block**: **Terminate/End Flow** (under "Terminate")
  - **Description**: Ends the contact when the interaction is complete.
  - **Settings**: 
    - No specific settings required.

---

# Example Flow

1. **Start Block** →  
2. **Play Prompt** (Greet the customer) →  
3. **Get Customer Input** (Gather input) →  
4. **Check Contact Attributes** (Check customer selection) →  
5. **Branch Logic** (For different customer needs) →  
6. **Transfer to Queue** (Route to the appropriate team) →  
7. **Check Queue Metrics** (Handle waiting) →  
8. **Set Customer Queue Flow** (Optional - queue experience) →  
9. **Terminate/End Flow** (End the call).

---

# Other Useful Blocks
- **Set Attributes** (under "Set"): Assign custom attributes to the contact for later use.
- **Loop** (under "Logic"): Repeat certain actions or conditions.
- **Invoke AWS Lambda Function** (under "Integrate"): Use for complex logic or external API calls.

-------

