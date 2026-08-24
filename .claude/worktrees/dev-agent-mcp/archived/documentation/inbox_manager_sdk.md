# Inbox Manager SDK Documentation

This documentation covers the Gmail integration functionality, which allows for automated email management through the Gmail API.

## Core Components

### GmailClient

The `GmailClient` class provides a robust interface for interacting with Gmail, handling authentication and common email operations.

```python
from npc.apis.gmail_client import GmailClient, GmailSearchOptions, Email

# Initialize the client
gmail_client = GmailClient()
```

#### Key Features

- Automatic authentication handling
- Retry mechanism with exponential backoff
- Error handling for common Gmail API issues

#### Search Options

The `GmailSearchOptions` class allows filtering email search results:

```python
search_options = GmailSearchOptions(
    labels=["Important", "Work"],  # Optional list of labels
    unread=True  # Optional boolean to filter by read status
)

# Retrieve threads matching criteria
threads = gmail_client.retrieve_threads(search_options, max_results=10)
```

#### Email Operations

```python
# Get the latest email from a thread
latest_email = gmail_client.get_latest_email(thread)

# Label management
gmail_client.apply_label(thread, "Important")
gmail_client.remove_label(thread, "Inbox")

# Thread management
gmail_client.archive_thread(thread)
gmail_client.mark_as_read(thread)
```

#### Email Object

The `Email` class represents an email message with the following attributes:

- `id`: Unique identifier
- `thread_id`: Thread identifier
- `subject`: Email subject
- `sender`: Sender's email address
- `recipient`: Recipient's email address
- `timestamp`: DateTime of the email
- `body`: Full email content
- `snippet`: Brief preview of the content

### InboxManager

The `InboxManager` class provides high-level email management functionality with AI-powered organization capabilities.

```python
from npc.simulators.ai_assistant.tools.inbox_manager import InboxManager
from npc.apis.llm_client import Model

# Initialize dependencies
llm = Model(...)  # Your LLM implementation
email_summarizer = EmailSummarizer(...)
gmail_client = GmailClient()

# Label mapping configuration
label_mapping = {
    EmailDestination.NEWSLETTER: "Newsletters",
    EmailDestination.BUSINESS_TRANSACTION: "Business",
    EmailDestination.ARCHIVE: "Archive",
    EmailDestination.DELETE: "Trash",
    EmailDestination.INBOX: "INBOX"
}

# Initialize manager
inbox_manager = InboxManager(
    llm=llm,
    email_summarizer=email_summarizer,
    gmail_client=gmail_client,
    label_mapping=label_mapping
)
```

#### Email Processing

```python
# Get threads to process
threads = gmail_client.retrieve_threads(GmailSearchOptions())

# Generate suggested actions
actions = inbox_manager.suggest_actions(threads)

# Execute actions
executed_actions = inbox_manager.execute_actions(actions)
```

### Email Destinations

The `EmailDestination` enum defines possible destinations for emails:

```python
from npc.prompts.ai_assistant.email_action_template import EmailDestination

# Available destinations
EmailDestination.NEWSLETTER          # For newsletters and subscriptions
EmailDestination.BUSINESS_TRANSACTION # For receipts, invoices, statements
EmailDestination.ARCHIVE            # For general archiving
EmailDestination.DELETE             # For spam, marketing, outdated content
EmailDestination.INBOX              # For important/immediate attention items
```

## Error Handling

The integration includes several error types for robust error handling:

- `GmailClientError`: Base exception for Gmail operations
- `AuthenticationError`: Raised for authentication failures
- `OperationError`: Raised for failed Gmail operations

```python
from npc.apis.gmail_client import GmailClientError, AuthenticationError, OperationError

try:
    gmail_client.retrieve_threads(search_options)
except AuthenticationError:
    # Handle authentication failure
except OperationError:
    # Handle operation failure
except GmailClientError:
    # Handle other Gmail-related errors
