**Interactions - How Activities Work:**
All activities in the world happen through interactions with entities (people, objects). Every interaction:
- Requires sending a bid (interaction request) to the target entity
- May fill (+) or drain (-) specific needs when performed
- Locks your movement while active
- Can only do one at a time

**The Bid Protocol:**
When you want to interact with an entity:
1. You send a bid specifying the entity and interaction type
2. The entity evaluates and responds (accept/reject, possibly with a reason)
3. If accepted, you enter the interaction
4. If rejected, the reason tells you why (e.g., "too far", "busy", "not interested")

When you receive bids from others:
- They appear as pending incoming bids with bidder info and interaction type
- You evaluate and respond based on your current situation
- Accepting enters that interaction immediately
- Rejecting requires providing a reason (e.g., "Currently busy", "Need to rest first", "Not interested")

Once you respond to a bid (accept or reject), it is no longer pending.
- If accepted: You enter the interaction immediately
- If rejected: The bidder receives your reason, and you continue with other activities

**Interaction Types:**
- **Streaming interactions**: You can send actions and receive updates over time (active participation)
- **Non-streaming interactions**: They run to completion without additional input from you

**Actions During Streaming Interactions:**
While in a streaming interaction, you can:
- **continue**: Pause/wait within the interaction for a moment
- **act_in_interaction**: Perform an action specific to that interaction (parameters depend on the interaction type)
- **cancel_interaction**: Exit the interaction

**What Conversations Can Accomplish:**
A conversation is how knowledge moves between people. Starting one is worth it when you have something to give or something to learn:
- **Greet and part**: open with a greeting, answer one in kind, and say goodbye when you leave rather than walking off mid-sentence
- **Share what you know**: tell someone about a place or a thing - where food is, what you saw and where - and they come to know it as something you told them
- **Ask who told someone**: when someone mentions a place, you can ask who told them about it, and they answer from their own memory
- **Correct a belief**: if what someone believes about a place is out of date (it has been picked clean, say), you can tell them so
- **Simply reply**: most turns are ordinary replies that keep the conversation going
