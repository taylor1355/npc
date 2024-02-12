import heapq

class EventQueue:
    def __init__(self, events=None):
        self.events = []
        if events is not None:
            self.push_batch(events)

    def push_batch(self, events):
        for event in events:
            self.push(event)

    def push(self, event):
        heapq.heappush(self.events, (event.timestep, event))

    def pop(self):
        _, event = heapq.heappop(self.events)
        return event

    def is_empty(self):
        return len(self.events) == 0

class Event:
    num_created_events = 0
    def __init__(self, timestep, metadata=None):
        self.timestep = timestep
        self.metadata = metadata
        self.id = Event.num_created_events
        Event.num_created_events += 1

class ChooseActionEvent(Event):
    def __init__(self, timestep, agent_id):
        super().__init__(timestep)
        self.agent_id = agent_id

class ConstructActionEvent(Event):
    def __init__(self, timestep, agent_id, action_intent):
        super().__init__(timestep)
        self.agent_id = agent_id
        self.action_intent = action_intent

class AffectRoomEvent(Event):
    def __init__(self, timestep, room_id, state_updating_code, check_state=False, metadata=None):
        super().__init__(timestep, metadata=metadata)
        self.room_id = room_id
        self.state_modifying_code = state_updating_code
        self.check_state = check_state

class CheckRoomStateEvent(Event):
    def __init__(self, timestep, room_id):
        super().__init__(timestep)
        self.room_id = room_id