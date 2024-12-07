import logging
from cmd import Cmd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from npc.apis.gmail_client import Email, GmailClient, GmailThread
from npc.apis.llm_client import LLMClient, Model
from npc.prompts.ai_assistant.inbox_management_template import (
    EmailDestination,
    prompt as inbox_management_prompt,
)
from npc.simulators.ai_assistant.tools.email_summarizer import EmailSummarizer
from npc.simulators.ai_assistant.tools.tool import Tool


class Status(Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


@dataclass
class InboxAction:
    """Action to be taken on an email thread"""
    thread: GmailThread
    destination: EmailDestination
    mark_as_read: bool
    original_destination: EmailDestination = field(init=False)
    original_mark_as_read: bool = field(init=False)
    mark_as_read_override: Optional[bool] = None
    status: Status = Status.NOT_STARTED

    def __post_init__(self):
        """Store original values for tracking modifications"""
        self.original_destination = self.destination
        self.original_mark_as_read = self.mark_as_read

    @property
    def should_mark_read(self) -> bool:
        """Determine if thread should be marked as read based on original or override value"""
        return (self.mark_as_read_override 
                if self.mark_as_read_override is not None 
                else self.mark_as_read)

    @property
    def is_modified(self) -> bool:
        """Check if the action has been modified from its original state"""
        return (self.mark_as_read != self.original_mark_as_read or 
                self.destination != self.original_destination)
    
    def execute(self, gmail_client: GmailClient, label_mapping: dict[str, str]) -> None:
        """Execute the action"""
        if self.status in (Status.SUCCEEDED, Status.CANCELED):
            return
        self.status = Status.IN_PROGRESS

        latest_email = None
        try:
            latest_email = gmail_client.get_latest_email(self.thread)
            if not latest_email:
                self.status = Status.FAILED
                return
                
            email_description = f"'{latest_email.subject}' from '{latest_email.sender}'"
            
            if self.destination != EmailDestination.INBOX:
                label = label_mapping[self.destination]
                if label:
                    gmail_client.apply_label(self.thread, label)
                gmail_client.archive_thread(self.thread)
                logging.info(f"Moved email {email_description} to {self.destination.value}")
            
            if self.should_mark_read:
                gmail_client.mark_as_read(self.thread)
                logging.info(f"Marked email {email_description} as read")
            
            self.status = Status.SUCCEEDED
            
        except Exception as e:
            self.status = Status.FAILED
            if not latest_email:
                email_description = f"from thread '{self.thread.id}'"
            logging.error(f"Failed to execute action for email '{email_description}': {repr(e)}")


class InboxManagerShell(Cmd):
    """Interactive command processor for managing email threads"""
    
    BATCH_SIZE = 5

    def __init__(self, threads: list[GmailThread], actions: list[InboxAction], gmail_client: GmailClient):
        super().__init__()
        self.threads = threads
        self.actions = actions
        self.gmail_client = gmail_client
        self.batch_start = 0
        self.completed = False
        self.canceled = False
        
        # Override cmd.Cmd settings
        self.prompt = "\nEnter command: "
        self.intro = "\nEmail Management Review\n" + "=" * 50
    
    def preloop(self) -> None:
        """Show initial batch of threads before starting command loop"""
        self.do_show("")
    
    def postcmd(self, stop: bool, line: str) -> bool:
        """Check if we should exit the command loop"""
        return stop or self.completed or self.canceled

    @property
    def current_batch_range(self) -> tuple[int, int]:
        """Get the start and end indices for the current batch"""
        if not self.threads:
            return (0, 0)
        return (
            self.batch_start, 
            min(self.batch_start + self.BATCH_SIZE, len(self.threads))
        )

    def _parse_thread_index(self, num_str: str) -> Optional[int]:
        """Convert user-provided number to thread index and validate"""
        try:
            idx = int(num_str) - 1
            start_idx, end_idx = self.current_batch_range
            batch_size = end_idx - start_idx
            
            if 0 <= idx < batch_size:
                return start_idx + idx
            print("Thread number out of range")
        except ValueError:
            print("Please provide a valid thread number")
        return None

    def do_read(self, arg: str) -> bool:
        """Toggle read status for a thread: read <number>"""
        if not arg:
            print("Usage: read <number>")
            return False
            
        action_idx = self._parse_thread_index(arg)
        if action_idx is not None:
            self.actions[action_idx].mark_as_read = not self.actions[action_idx].mark_as_read
            self.do_show("")
        return False

    def do_move(self, arg: str) -> bool:
        """Change destination for a thread: move <number> <destination>"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: move <number> <destination>")
            return False
            
        action_idx = self._parse_thread_index(args[0])
        if action_idx is not None:
            try:
                new_dest = EmailDestination(args[1].upper())
                self.actions[action_idx].destination = new_dest
                self.do_show("")
            except ValueError:
                print(f"Valid destinations: {EmailDestination.choices_str()}")
        return False

    def do_cancel(self, arg: str) -> bool:
        """Cancel actions for specified threads: cancel <number> [number2...]"""
        if not arg:
            print("Usage: cancel <number> [number2...]")
            return False
            
        for num in arg.split():
            action_idx = self._parse_thread_index(num.strip())
            if action_idx is not None:
                self.actions[action_idx].status = Status.CANCELED
        
        self.do_show("")
        return False

    def do_next(self, arg: str) -> bool:
        """Show next batch of threads"""
        if self.batch_start + self.BATCH_SIZE < len(self.threads):
            self.batch_start += self.BATCH_SIZE
            self.do_show("")
        else:
            print("No more threads")
        return False

    def do_prev(self, arg: str) -> bool:
        """Show previous batch of threads"""
        if self.batch_start > 0:
            self.batch_start = max(0, self.batch_start - self.BATCH_SIZE)
            self.do_show("")
        else:
            print("Already at first batch")
        return False

    def do_show(self, arg: str) -> bool:
        """Display current batch of threads"""
        start_idx, end_idx = self.current_batch_range
        current_batch = list(zip(
            self.threads[start_idx:end_idx],
            self.actions[start_idx:end_idx]
        ))
        
        for i, (thread, action) in enumerate(current_batch, 1):
            latest_email = self.gmail_client.get_latest_email(thread)
            if latest_email:
                canceled_str = "[ACTION CANCELED] " if action.status == Status.CANCELED else ""
                print(f"\n{i}. {canceled_str}From: {latest_email.sender}")

                print(f"   Subject: {latest_email.subject}")

                destination_str = f"Move to {action.destination.value}"
                read_str = " and mark as read" if action.mark_as_read else ""
                modified_suffix = "*" if action.is_modified else ""
                print(f"   Action: {destination_str}{read_str}{modified_suffix}")
        
        self._show_navigation_hints(end_idx)
        return False

    def do_done(self, arg: str) -> bool:
        """Complete review and execute actions"""
        self.completed = True
        return True

    def do_quit(self, arg: str) -> bool:
        """Cancel all actions and exit"""
        self.canceled = True
        return True

    def _show_navigation_hints(self, end_idx: int) -> None:
        """Show available navigation commands"""
        print("\nCommands:")
        print("read <number>     - Toggle read status")
        print("move <number> <destination> - Change destination")
        print("cancel <number>   - Cancel action")
        if self.batch_start > 0:
            print("prev            - Previous batch")
        if end_idx < len(self.threads):
            print("next            - Next batch")
        print("show            - Show current batch")
        print("done            - Complete review")
        print("quit            - Cancel all and exit")
        print("help            - Show this message")


class InboxManager(Tool):
    """Tool for managing inbox by automatically suggesting and executing actions on email threads"""
    def __init__(
        self,
        llm: Model,
        email_summarizer: EmailSummarizer,
        gmail_client: GmailClient,
        label_mapping: dict[EmailDestination, str]
    ) -> None:
        super().__init__(llm)
        self.email_summarizer = email_summarizer
        self.gmail_client = gmail_client

        self.label_mapping = label_mapping
        for dest in EmailDestination:
            if dest not in self.label_mapping:
                raise ValueError(f"Incomplete label mapping, missing a {dest} label")

    def _initialize_generators(self) -> None:
        """Initialize the LLM client for generating inbox management decisions"""
        self.inbox_manager = LLMClient(inbox_management_prompt, self.llm)

    def manage_inbox(self, threads: list[GmailThread]) -> list[InboxAction]:
        return self.execute(threads)
    
    def execute(self, threads: list[GmailThread]) -> list[InboxAction]:
        """Suggest and execute actions for organizing the inbox with user interaction"""
        if not threads:
            logging.info("No threads to process")
            return []
            
        actions = self._suggest_actions(threads)
        
        # Use command shell for interaction
        shell = InboxManagerShell(threads, actions, self.gmail_client)
        shell.cmdloop()
        
        if shell.canceled:
            return []
            
        # Execute non-canceled actions
        for action in actions:
            if action.status != Status.CANCELED:
                action.execute(self.gmail_client, self.label_mapping)
        
        # TODO: print summary of executed actions
        # Also print failed actions, if any

        return [action for action in actions if action.status == Status.SUCCEEDED]
    
    def _suggest_actions(self, threads: list[GmailThread]) -> list[InboxAction]:
        """
        Generate suggested actions for the given threads
        
        Args:
            threads: List of email threads to process
            
        Returns:
            List of suggested actions
        """
        actions: list[InboxAction] = []
        for thread in threads:
            try:
                # Get latest email from thread
                latest_email = self.gmail_client.get_latest_email(thread)
                if not latest_email:
                    continue
                
                # Get or generate summary
                summary = self.email_summarizer.summarize(latest_email)
                
                # Generate inbox management suggestion
                response = self.inbox_manager.generate_response(
                    email_summary=summary.summary,
                    sender=latest_email.sender,
                    subject=latest_email.subject
                )
                
                action = InboxAction(
                    thread=thread,
                    destination=response["destination"],
                    mark_as_read=response["mark_as_read"].lower() == "true",
                )
                actions.append(action)
            except Exception as e:
                logging.error(f"Failed to suggest action for email from {latest_email.sender if latest_email else 'unknown'}")
                # Create default action that keeps thread in inbox
                action = InboxAction(
                    thread=thread,
                    destination=EmailDestination.INBOX,
                    mark_as_read=False,
                )
                actions.append(action)
        
        return actions
    
    @property
    def description(self) -> str:
        return "Manages inbox by organizing email threads based on content analysis and user preferences"
    
    @property
    def required_inputs(self) -> dict[str, str]:
        return {
            "threads": "List of GmailThread objects to process"
        }
