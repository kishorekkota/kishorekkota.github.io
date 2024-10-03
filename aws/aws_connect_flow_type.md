# Amazon Connect Contact Flow Types

Amazon Connect offers various types of **Contact Flows** tailored for different purposes. Each flow type manages specific parts of the customer journey.

## 1. Inbound Contact Flow
- **Purpose**: Manages incoming customer calls, routing them based on customer input or data.
- **Use Case**: Handles interactions from greeting the customer to transferring them to an appropriate queue.
- **Key Blocks**:
  - **Play Prompt** (greeting)
  - **Get Customer Input** (IVR options)
  - **Check Contact Attributes** (decision logic)
  - **Transfer to Queue** (queue routing)

## 2. Outbound Contact Flow
- **Purpose**: Manages automated outbound calls from the contact center to customers.
- **Use Case**: Used for proactive customer outreach such as appointment reminders, surveys, or marketing calls.
- **Key Blocks**:
  - **Call Customer**
  - **Play Prompt**
  - **Invoke AWS Lambda Function** (for customer-specific data)
  - **Terminate** (call end)

## 3. Customer Queue Flow
- **Purpose**: Handles the customer's experience while they are waiting in a queue.
- **Use Case**: Triggered when the customer is placed on hold or waiting for an agent.
- **Key Blocks**:
  - **Play Prompt** (hold music, wait-time announcements)
  - **Check Queue Metrics** (e.g., wait time, agent availability)
  - **Loop** (repeat messages/music until agent available)

## 4. Agent Whisper Flow
- **Purpose**: Delivers information to the agent right before answering a call.
- **Use Case**: Provides additional info (e.g., customer details) to the agent without the customer hearing it.
- **Key Blocks**:
  - **Set Contact Attributes**
  - **Play Prompt** (to agent only)
  - **Transfer to Agent**

## 5. Hold Flow
- **Purpose**: Manages the experience when an agent places the customer on hold.
- **Use Case**: Ensures the customer hears music or messages while waiting on hold.
- **Key Blocks**:
  - **Play Prompt** (hold music or announcements)
  - **Loop** (repeats until agent takes customer off hold)

## 6. Transfer to Agent Flow
- **Purpose**: Manages what happens when a contact is transferred from one agent to another.
- **Use Case**: Dictates how a transfer to another agent or department is handled.
- **Key Blocks**:
  - **Transfer to Queue** (route to another agent)
  - **Set Contact Attributes** (pass information during transfer)

## 7. Transfer to Queue Flow
- **Purpose**: Handles the logic for transferring the contact to a specific queue.
- **Use Case**: Directs a customer to a different queue for handling.
- **Key Blocks**:
  - **Check Contact Attributes** (determine the appropriate queue)
  - **Transfer to Queue**

## 8. Voicemail Flow
- **Purpose**: Manages the interaction when a customer is sent to leave a voicemail.
- **Use Case**: If no agents are available, directs customers to leave a message for follow-up.
- **Key Blocks**:
  - **Play Prompt** (voicemail instructions)
  - **Store Customer Recording** (save voicemail)
  - **Terminate** (end the call after voicemail)

## 9. Error Handling Flow
- **Purpose**: Manages the flow when an error occurs (e.g., unrecognized input, system error).
- **Use Case**: Triggered if something goes wrong during the interaction, like incorrect IVR input.
- **Key Blocks**:
  - **Play Prompt** (error message)
  - **Transfer to Agent** (for manual intervention)
  - **Terminate** (if necessary)

## 10. Lex Bot Flow
- **Purpose**: Integrates with an Amazon Lex bot for voice or chat interactions.
- **Use Case**: For advanced AI-driven conversations using natural language understanding (NLU).
- **Key Blocks**:
  - **Invoke AWS Lambda Function** (fetch customer data)
  - **Interact with Lex Bot** (dynamic conversation)
  - **Check Contact Attributes** (route based on bot interactions)

---

## Summary Table of Flow Types

| Flow Type              | Purpose                                          | Example Use Case                                   |
|------------------------|--------------------------------------------------|---------------------------------------------------|
| Inbound Contact Flow    | Manages incoming calls                          | Routing customer calls to appropriate agents      |
| Outbound Contact Flow   | Manages outbound calls                          | Automated appointment reminders                   |
| Customer Queue Flow     | Handles customer experience in queue            | Playing music while customers wait                |
| Agent Whisper Flow      | Delivers info to agents                         | Providing customer data to agents before calls    |
| Hold Flow               | Manages experience when placed on hold          | Playing hold music or announcements               |
| Transfer to Agent Flow  | Manages agent-to-agent transfers                | Moving a customer from one agent to another       |
| Transfer to Queue Flow  | Transfers customer to specific queue            | Routing to a different department                 |
| Voicemail Flow          | Manages voicemail                               | Directing customers to leave a voicemail          |
| Error Handling Flow     | Handles errors or invalid inputs                | Providing fallback prompts or routing options     |
| Lex Bot Flow            | Integrates with Amazon Lex                      | AI-driven conversations using voice or chatbots   |

---

Each of these flow types can be customized to fit your contact center’s needs. You can mix and match these flows to create a seamless customer experience.


---



# Inbound Contact Flow Blocks (In Order)

For an **Inbound Contact Flow** in Amazon Connect, the blocks you need to connect typically follow a logical progression from greeting the customer, collecting input, routing them to the appropriate queue, and finally ending the call. Below is the list of blocks in the specific order they are typically used:

## Inbound Contact Flow Blocks (In Order)

1. **Start Block**
   - The entry point for the flow.

2. **Play Prompt**
   - Greets the customer (e.g., “Welcome to [Your Company]”).

3. **Get Customer Input** (Optional)
   - Collects input from the customer via DTMF (keypad) or speech. (e.g., “Press 1 for Sales, Press 2 for Support”).

4. **Check Contact Attributes**
   - Evaluates customer input or existing attributes to determine routing decisions (e.g., if “1” was pressed, send to Sales).

5. **Branch** (Optional)
   - Additional decision logic based on customer attributes (e.g., routing based on customer type or other conditions).

6. **Transfer to Queue**
   - Routes the customer to the appropriate queue based on their input (e.g., Sales or Support queue).

7. **Set Customer Queue Flow** (Optional)
   - Handles what happens while the customer is waiting in the queue (e.g., hold music, estimated wait time).

8. **Check Queue Metrics** (Optional)
   - Checks real-time queue information, such as the number of available agents or estimated wait time, to make further routing decisions.

9. **Loop** (Optional)
   - Used to repeat hold messages or music while waiting in the queue.

10. **Transfer to Agent**
    - Connects the customer with an available agent from the assigned queue.

11. **Terminate/End Flow**
    - Ends the call or flow when it completes (e.g., after the call with the agent).

---

## Example Sequence:

1. **Start Block**
2. **Play Prompt** (Greeting)
3. **Get Customer Input** (IVR options)
4. **Check Contact Attributes** (Evaluate input)
5. **Branch** (Logic decision-making)
6. **Transfer to Queue** (Route based on selection)
7. **Set Customer Queue Flow** (While waiting)
8. **Check Queue Metrics** (Optional for dynamic routing)
9. **Loop** (Optional for holding)
10. **Transfer to Agent**
11. **Terminate/End Flow**

You can adjust these steps depending on your specific use case, but this order generally works for most inbound call scenarios.


---

# Outbound Contact Flow Blocks (In Order)

For an **Outbound Contact Flow** in Amazon Connect, the flow is designed to handle calls initiated by the contact center, such as appointment reminders, surveys, or marketing calls. Below is the list of blocks in the specific order they are typically used:

## Outbound Contact Flow Blocks (In Order)

1. **Start Block**
   - The entry point for the flow.

2. **Call Customer**
   - Initiates the outbound call to the customer.

3. **Check Contact Attributes** (Optional)
   - Evaluates any customer-specific attributes that may impact the call (e.g., checking customer type, contact preferences, etc.).

4. **Set Contact Attributes** (Optional)
   - Sets any attributes needed for the outbound call, such as customer data or any custom parameters.

5. **Play Prompt**
   - Plays a message to the customer once the call is connected (e.g., “Hello, this is a reminder from [Your Company]”).

6. **Get Customer Input** (Optional)
   - If needed, collects input from the customer via DTMF (keypad) or speech (e.g., “Press 1 to confirm your appointment, Press 2 to speak with an agent”).

7. **Check Contact Attributes** (Optional)
   - Evaluates the customer’s input to determine further action (e.g., if “1” is pressed, mark appointment confirmed).

8. **Branch** (Optional)
   - Provides decision logic based on customer input or attributes (e.g., send them to an agent if they need more help).

9. **Invoke AWS Lambda Function** (Optional)
   - Invokes a Lambda function for backend processing, such as updating a database with customer responses or retrieving more customer data.

10. **Transfer to Queue** (Optional)
    - Routes the call to a queue if the customer chooses to speak with an agent.

11. **Set Customer Queue Flow** (Optional)
    - Handles the experience while the customer is waiting in the queue, such as playing hold music or announcements.

12. **Terminate/End Flow**
    - Ends the call once the flow completes, either after the interaction with an agent or after delivering the outbound message.

---

## Example Sequence:

1. **Start Block**
2. **Call Customer** (Initiate the outbound call)
3. **Check Contact Attributes** (Evaluate customer data)
4. **Set Contact Attributes** (Optional settings for call)
5. **Play Prompt** (Deliver the message)
6. **Get Customer Input** (If interaction is needed)
7. **Check Contact Attributes** (Evaluate the response)
8. **Branch** (Make decisions based on customer input)
9. **Invoke AWS Lambda Function** (Backend processing, if needed)
10. **Transfer to Queue** (If the customer needs to talk to an agent)
11. **Set Customer Queue Flow** (Handle the experience while waiting)
12. **Terminate/End Flow** (End the call)

---

This order provides a basic structure for handling outbound customer calls efficiently, ensuring all key touchpoints are covered.



---

# Customer Queue Flow Blocks (In Order)

A **Customer Queue Flow** in Amazon Connect manages the customer experience while they are waiting in a queue, often providing information or hold music. Below is the list of blocks in the order they are typically connected.

## Customer Queue Flow Blocks (In Order)

1. **Start Block**
   - The entry point for the flow.

2. **Set Logging Behavior** (Optional)
   - Determines logging settings for tracking customer progress through the flow.

3. **Play Prompt**
   - Plays hold music or messages to the customer (e.g., “Thank you for holding. We will connect you with the next available agent.”).

4. **Check Queue Metrics**
   - Retrieves real-time information about the queue, such as the number of waiting customers or the estimated wait time. This can be used to determine further actions.

5. **Loop**
   - Repeats the hold message or music until certain conditions are met (e.g., until the customer is connected to an agent).

6. **Check Queue Status** (Optional)
   - Checks if the customer should continue waiting or if further action is required, such as offering a callback option if the wait time is too long.

7. **Set Customer Queue Flow**
   - Configures what happens while the customer is waiting, such as periodic announcements or updated wait times.

8. **Play Prompt** (Optional)
   - Plays additional prompts or updates based on the real-time queue status, such as informing the customer of their position in the queue.

9. **Transfer to Agent**
   - Transfers the customer to the next available agent in the queue.

10. **Terminate/End Flow**
    - Ends the flow if the customer chooses to end the call or if the interaction is otherwise completed.

---

## Example Sequence:

1. **Start Block**
2. **Set Logging Behavior** (Optional logging)
3. **Play Prompt** (Hold music or message)
4. **Check Queue Metrics** (Evaluate queue conditions)
5. **Loop** (Repeat hold music/message)
6. **Check Queue Status** (Optional dynamic decisions)
7. **Set Customer Queue Flow** (Handle customer waiting experience)
8. **Play Prompt** (Optional updates or announcements)
9. **Transfer to Agent** (Connect to an available agent)
10. **Terminate/End Flow** (End the flow)

---

This flow ensures that customers waiting in the queue have a smooth experience, with periodic updates, hold music, and the opportunity to connect with an agent when available.


---

# Agent Whisper Flow Blocks (In Order)

An **Agent Whisper Flow** in Amazon Connect is used to deliver a message to the agent just before connecting them with a customer. This message can include important details about the caller or instructions for the agent. Below is the list of blocks in the specific order they are typically connected.

## Agent Whisper Flow Blocks (In Order)

1. **Start Block**
   - The entry point for the flow.

2. **Check Contact Attributes** (Optional)
   - Retrieves and evaluates any relevant customer information or contact attributes that can help tailor the whisper message.

3. **Set Contact Attributes** (Optional)
   - Used to set or update any contact attributes that might be useful for the agent during the interaction.

4. **Play Prompt**
   - Plays a message to the agent, such as details about the customer or special instructions (e.g., “This customer is a VIP, please offer them premium support”).

5. **Invoke AWS Lambda Function** (Optional)
   - Calls a Lambda function to retrieve more detailed information or process backend logic before the agent is connected.

6. **Loop** (Optional)
   - If necessary, repeats a message or updates until certain conditions are met (such as retrieving complete customer data).

7. **Transfer to Agent**
   - Connects the agent to the customer once the whisper message is completed.

8. **Terminate/End Flow**
   - Ends the whisper flow after the agent is connected to the customer.

---

## Example Sequence:

1. **Start Block**
2. **Check Contact Attributes** (Optional retrieval of customer info)
3. **Set Contact Attributes** (Optional for updating agent's context)
4. **Play Prompt** (Deliver whisper message to the agent)
5. **Invoke AWS Lambda Function** (Optional backend processing)
6. **Loop** (Optional for repetition)
7. **Transfer to Agent** (Connect agent to customer)
8. **Terminate/End Flow** (End the whisper flow)

---

This flow helps ensure that agents receive crucial information about the caller or special instructions before engaging with the customer, improving the overall call quality and efficiency.


---

# Hold Flow Blocks (In Order)

A **Hold Flow** in Amazon Connect manages the experience for customers who are placed on hold while waiting for an agent. This flow typically includes playing music or messages to keep the customer engaged. Below is the list of blocks in the specific order they are typically connected.

## Hold Flow Blocks (In Order)

1. **Start Block**
   - The entry point for the flow.

2. **Play Prompt**
   - Plays hold music or messages to the customer (e.g., “Thank you for holding. Your call is important to us.”).

3. **Set Customer Queue Flow** (Optional)
   - Configures what will happen while the customer is waiting, including any announcements about estimated wait times or updates.

4. **Check Queue Metrics** (Optional)
   - Evaluates real-time queue metrics to decide if further action is needed, such as informing the customer of their position in the queue.

5. **Loop**
   - Repeats the hold music or messages until the customer is connected to an agent or decides to hang up.

6. **Check Queue Status** (Optional)
   - Checks whether the customer should continue waiting or if other options should be presented (like a callback).

7. **Transfer to Agent**
   - Connects the customer to the next available agent once one becomes available.

8. **Terminate/End Flow**
   - Ends the hold flow when the call with the agent is established or if the interaction is otherwise completed.

---

## Example Sequence:

1. **Start Block**
2. **Play Prompt** (Hold music or message)
3. **Set Customer Queue Flow** (Optional for customer experience)
4. **Check Queue Metrics** (Optional for real-time updates)
5. **Loop** (Repeat music/messages)
6. **Check Queue Status** (Optional for dynamic decisions)
7. **Transfer to Agent** (Connect customer to agent)
8. **Terminate/End Flow** (End the hold flow)

---

This flow ensures that customers on hold receive appropriate music or messages to enhance their waiting experience while they are connected to an agent.

---

# Transfer to Agent Flow Blocks (In Order)

A **Transfer to Agent Flow** in Amazon Connect facilitates the process of connecting a customer to a live agent after initial interaction, whether through an IVR or a queue. Below is the list of blocks in the specific order they are typically connected.

## Transfer to Agent Flow Blocks (In Order)

1. **Start Block**
   - The entry point for the flow.

2. **Check Queue Metrics** (Optional)
   - Evaluates the status of the queue, such as the number of available agents, estimated wait times, or customer position in the queue.

3. **Set Customer Queue Flow** (Optional)
   - Configures the waiting experience while the customer is being transferred, including hold music or informational messages.

4. **Play Prompt** (Optional)
   - Delivers a message to the customer before transfer (e.g., “You will be connected to an agent shortly. Please hold.”).

5. **Check Contact Attributes** (Optional)
   - Retrieves relevant customer data or attributes that may affect the transfer process (e.g., preferred agent type or language).

6. **Branch** (Optional)
   - Provides decision logic based on the customer's attributes or input (e.g., transferring to a specialized agent based on customer needs).

7. **Transfer to Queue** (Optional)
   - Places the customer in a specific queue for agents who can assist with their issue.

8. **Transfer to Agent**
   - Connects the customer to the next available agent from the appropriate queue.

9. **Terminate/End Flow**
   - Ends the flow once the transfer to the agent is successful or if the interaction is otherwise completed.

---

## Example Sequence:

1. **Start Block**
2. **Check Queue Metrics** (Optional for evaluating agent availability)
3. **Set Customer Queue Flow** (Optional for managing the waiting experience)
4. **Play Prompt** (Optional for informing the customer)
5. **Check Contact Attributes** (Optional for retrieving customer data)
6. **Branch** (Optional for decision-making based on attributes)
7. **Transfer to Queue** (Optional for specific routing)
8. **Transfer to Agent** (Connect customer to an agent)
9. **Terminate/End Flow** (End the transfer flow)

---

This flow ensures a smooth transition for customers from automated processes or queues to speaking with a live agent, enhancing the overall customer experience.

---

# Transfer to Queue Flow Blocks (In Order)

A **Transfer to Queue Flow** in Amazon Connect directs customers to a specific queue based on their needs after they have interacted with an IVR system or agent. This flow helps ensure that customers are routed to the appropriate team for assistance. Below is the list of blocks in the specific order they are typically connected.

## Transfer to Queue Flow Blocks (In Order)

1. **Start Block**
   - The entry point for the flow.

2. **Check Contact Attributes** (Optional)
   - Retrieves any relevant customer data or attributes that may affect the transfer process, such as customer type or issue category.

3. **Play Prompt** (Optional)
   - Delivers a message to the customer before the transfer (e.g., “You will be transferred to the appropriate department. Please hold.”).

4. **Check Queue Metrics** (Optional)
   - Evaluates the status of the target queue, such as the number of available agents, estimated wait times, or the customer's position in the queue.

5. **Set Customer Queue Flow** (Optional)
   - Configures the waiting experience while the customer is in the queue, including hold music or announcements.

6. **Branch** (Optional)
   - Provides decision logic based on customer attributes or input to determine the correct queue to transfer the customer to.

7. **Transfer to Queue**
   - Places the customer in the appropriate queue based on the logic established in previous blocks.

8. **Set Customer Queue Flow** (Optional)
   - Handles what happens while the customer is waiting in the new queue, such as playing additional hold music or messages.

9. **Transfer to Agent** (Optional)
   - Connects the customer to the next available agent from the assigned queue once an agent becomes available.

10. **Terminate/End Flow**
    - Ends the flow when the transfer is complete, either after connecting to an agent or if the interaction is otherwise completed.

---

## Example Sequence:

1. **Start Block**
2. **Check Contact Attributes** (Optional for retrieving customer data)
3. **Play Prompt** (Optional for informing the customer)
4. **Check Queue Metrics** (Optional for evaluating queue status)
5. **Set Customer Queue Flow** (Optional for managing the waiting experience)
6. **Branch** (Optional for decision-making based on attributes)
7. **Transfer to Queue** (Place the customer in the appropriate queue)
8. **Set Customer Queue Flow** (Optional for handling waiting experience)
9. **Transfer to Agent** (Optional for connecting to an agent)
10. **Terminate/End Flow** (End the transfer flow)

---

This flow ensures customers are effectively routed to the right queue for their needs, enhancing their overall experience and improving call handling efficiency.
