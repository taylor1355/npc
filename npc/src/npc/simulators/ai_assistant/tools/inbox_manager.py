import logging
from dataclasses import dataclass
from typing import List, Optional

from npc.apis.gmail_client import Email
from npc.apis.llm_client import LLMClient, Model
from npc.prompts.ai_assistant.inbox_management_template import (
    EmailDestination,
    prompt as inbox_management_prompt,
)
from npc.simulators.ai_assistant.tools.email_summarizer import EmailSummarizer
from npc.simulators.ai_assistant.tools.tool import Tool


@dataclass
class InboxAction:
    """Action to be taken on an email"""
    email_id: str
    thread_id: str
    destination: EmailDestination
    requires_attention: bool
    canceled: bool = False


class InboxManager(Tool):
    """Tool for managing inbox by organizing emails based on their content"""
    
    def __init__(self, llm: Model, email_summarizer: EmailSummarizer) -> None:
        """
        Initialize the inbox manager
        
        Args:
            llm: Language model for determining email destinations
            email_summarizer: Tool for generating email summaries
        """
        super().__init__(llm)
        self.email_summarizer = email_summarizer

    def _initialize_generators(self) -> None:
        """Initialize the LLM client for generating inbox management decisions"""
        self.inbox_manager = LLMClient(inbox_management_prompt, self.llm)

    def manage_inbox(self, emails: List[Email]) -> List[InboxAction]:
        return self.execute(emails)
    
    def execute(self, emails: List[Email]) -> List[InboxAction]:
        """
        Suggest and execute actions for organizing the inbox with user interaction
        
        Args:
            emails: List of emails to process
            
        Returns:
            List of executed actions
        """
        actions = self._suggest_actions(emails)
        if not self._interactive_flow(emails, actions):
            return []  # User canceled the flow
        
        # Get list of emails requiring attention
        attention_required = [
            (email, action) for email, action in zip(emails, actions)
            if action.requires_attention and not action.canceled
        ]
        
        # Send notification if there are emails requiring attention
        if attention_required:
            self._notify_user_of_important_emails(attention_required)

        # Execute actions that have not been canceled
        self._execute_actions(actions)
        
        return [action for action in actions if not action.canceled]
    
    def _suggest_actions(self, emails: List[Email]) -> List[InboxAction]:
        """
        Generate suggested actions for the given emails
        
        Args:
            emails: List of emails to process
            
        Returns:
            List of suggested actions
        """
        actions: List[InboxAction] = []
        for email in emails:
            # Get or generate summary
            summary = self.email_summarizer.execute(email)
            
            # Generate inbox management suggestion
            response = self.inbox_manager.generate_response(
                email_summary=summary.summary,
                sender=email.sender,
                subject=email.subject
            )
            
            action = InboxAction(
                email_id=email.id,
                thread_id=email.thread_id,
                destination=response['destination'],
                requires_attention=response['user_intervention_required'].lower() == 'true',
            )
            actions.append(action)
        
        return actions

    def _interactive_flow(self, emails: List[Email], actions: List[InboxAction]) -> bool:
        """
        Take the user through an interactive flow to review and confirm actions
        
        Args:
            emails: List of emails being processed
            actions: List of suggested actions
            
        Returns:
            bool: True if flow completed, False if canceled
        """
        print("\nSuggested Inbox Management Actions:")
        print("----------------------------------")
        
        # Display each email and its suggested action
        for i, (email, action) in enumerate(zip(emails, actions), 1):
            attention = "Requires Attention" if action.requires_attention else "No Action Required"
            print(f"\n{i}. Email from: {email.sender}")
            print(f"   Subject: {email.subject}")
            print(f"   Status: {attention}")
            print(f"   Destination: {action.destination}")
        
        print("\nOptions:")
        print("p - Proceed with all actions")
        print("c [numbers] - Cancel specific emails (e.g., 'c 1,3' cancels emails 1 and 3)")
        print("c - Cancel all actions")
        
        valid_choice = False
        while not valid_choice:
            choice = input("\nEnter your choice: ").strip().lower()
        
            if choice == "p":
                return True
            elif choice.startswith("c "):
                try:
                    # Extract numbers from input like "c 1,3,5" and convert to 0-based index
                    indices = [int(n.strip()) - 1 for n in choice[2:].split(",")]
                    if any(i < 0 or i >= len(actions) for i in indices):
                        print("\nInvalid email numbers to cancel. Please try again.")
                        continue
                    for i in indices:
                        actions[i].canceled = True
                    return True
                except ValueError:
                    print("\nInvalid list of emails to cancel. Expected format: 'c {number1},{number2},...'")
                    continue
            elif choice == "c":
                print("\nCanceling all actions.")
                return False
            else:
                print(f"\nInvalid choice {choice}. Please try again.")
                continue
    
    def _execute_actions(self, actions: List[InboxAction]) -> None:
        """
        Execute the actions that have not been canceled
        
        Args:
            actions: List of actions to execute
        """
        for action in actions:
            if action.canceled:
                logging.info(f"Skipping canceled action for email {action.email_id}")
                continue
                
            try:
                if action.destination == EmailDestination.KEEP_IN_INBOX:
                    # No action needed, email stays in inbox
                    continue
                    
                # TODO: take in label names as part of the InboxManager constructor rather than hardcoding them
                elif action.destination == EmailDestination.NEWSLETTER:
                    self.gmail_client.apply_label(action.thread_id, 'saved-emails-newsletters')
                    self.gmail_client.remove_label(action.thread_id, 'INBOX')
                    logging.info(f"Moved email {action.email_id} to Newsletter label")                    

                elif action.destination == EmailDestination.BUSINESS_TRANSACTION:
                    self.gmail_client.apply_label(action.thread_id, 'saved-emails-receipts')
                    self.gmail_client.remove_label(action.thread_id, 'INBOX')
                    logging.info(f"Moved email {action.email_id} to Business label")

                    
                elif action.destination == EmailDestination.ARCHIVE:
                    self.gmail_client.archive_thread(action.thread_id)
                    logging.info(f"Archived email {action.email_id}")
                    
                elif action.destination == EmailDestination.DELETE:
                    # Note: We archive instead of delete for safety
                    self.gmail_client.archive_thread(action.thread_id)
                    self.gmail_client.apply_label(action.thread_id, 'To-Delete')
                    logging.info(f"Marked email {action.email_id} for deletion")
                
                # Mark as read if no user attention required
                if not action.requires_attention:
                    self.gmail_client.mark_as_read(action.thread_id)
                    logging.info(f"Marked email {action.email_id} as read")
                
                
            except Exception as e:
                logging.error(f"Failed to execute action for email {action.email_id}: {str(e)}")

    def _notify_user_of_important_emails(self, attention_required: List[tuple[Email, InboxAction]]) -> None:
        """
        Send a text message notification about emails requiring attention
        
        Args:
            attention_required: List of (email, action) tuples for emails needing attention
        """
        # TODO: Integrate with SMS API service
        message = "Important emails requiring your attention:\n\n"
        
        for email, action in attention_required:
            message += f"From: {email.sender}\n"
            message += f"Subject: {email.subject}\n"
            message += f"Reason: {action.reasoning}\n\n"
        
        logging.info("Would send SMS with message:\n%s", message)
    
    @property
    def description(self) -> str:
        return "Manages inbox by organizing emails based on content analysis and user preferences"
    
    @property
    def required_inputs(self) -> dict[str, str]:
        return {
            "emails": "List of Email objects to process"
        }
