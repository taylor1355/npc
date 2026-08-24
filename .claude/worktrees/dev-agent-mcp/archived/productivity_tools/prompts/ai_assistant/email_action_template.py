from enum import Enum
from typing import Self
import logging

from npc.prompts.prompt_common import create_prompt_template, Prompt, TagPattern


# TODO: move to gmail_client.py
class EmailDestination(Enum):
    NEWSLETTER = "newsletter"
    BUSINESS_TRANSACTION = "business_transaction"
    ARCHIVE = "archive"
    DELETE = "delete"
    INBOX = "inbox"

    @classmethod
    def choices_str(cls) -> str:
        """Get a string containing all valid destination choices."""
        return "/".join(dest.name for dest in cls)

    def parse(text: str) -> Self:
        """Convert a string to an EmailDestination enum.

        Args:
            text: String representation of destination

        Returns:
            Corresponding EmailDestination enum value, defaults to INBOX if invalid
        """
        try:
            return EmailDestination[text.strip().upper()]
        except KeyError:
            logging.warning(f"Invalid destination value: {text}. Defaulting to INBOX.")
            return EmailDestination.INBOX


SYSTEM_PROMPT = [
    "You are an expert email organization assistant specializing in inbox management and workflow optimization."
    " Your task is to analyze an email and determine whether it should be marked as read and what destination"
    " it should be sent to."
]

USER_PROMPT = [
    "# Email Information",
    "Here is the email information you need to analyze:",
    "<sender>",
    "{sender}",
    "</sender>",
    "<subject>",
    "{subject}",
    "</subject>",
    "<email_summary>",
    "{email_summary}",
    "</email_summary>",
    "",
    "# Analysis Guidelines",
    "Please follow these steps to analyze the email:",
    "",
    "## 1. Read Status Assessment",
    "First, analyze whether the email should be marked as read. Consider the following factors:",
    "",
    "Keep Unread (Don't Mark as Read):",
    "- Personal communications from friends, family, or important contacts",
    "- Time-sensitive notifications (e.g., payment deadlines, appointment reminders)",
    "- Security-related communications (2FA codes, account alerts)",
    "- Direct requests or questions requiring a response",
    "- Important business communications needing review or action",
    "",
    "Mark as Read:",
    "- Newsletters and promotional content",
    "- Automated system notifications",
    "- Marketing materials",
    "- Routine updates or announcements",
    "- Informational emails not requiring action",
    "",
    "## 2. Destination Selection",
    "Then, determine the most appropriate destination for the email:",
    "",
    "NEWSLETTER:",
    "- Newsletters",
    "",
    "BUSINESS_TRANSACTION:",
    "- Receipts and invoices",
    "- Payment confirmations",
    "- Account statements",
    "- Terms of service updates",
    "- Privacy policy changes",
    "",
    "ARCHIVE:",
    "- Anything that does not match the other categories",
    "",
    "DELETE:",
    "- Spam",
    "- Marketing material or promotions",
    "- Outdated announcements",
    "- Duplicate messages",
    "- Temporary notifications (e.g., one-time codes)",
    "",
    "INBOX:",
    "- Emails that the user sent to themselves",
    "- Emails requiring immediate attention",
    "- Ongoing discussions",
    "- Pending tasks or decisions",
    "- Time-sensitive materials",
    "",
    "# Response Format",
    "Provide your analysis in the following structure:",
    "",
    "## 1. Detailed Analysis",
    "In <read_status_analysis> tags, provide a JSON-structured analysis:",
    "<read_status_analysis>",
    "{{",
    "   'keep_unread_reasons': 'Factors supporting keeping unread'",
    "   'mark_read_reasons': 'Factors supporting marking as read'",
    "}}",
    "</read_status_analysis>",
    "",
    "In <destination_analysis> tags, provide a JSON-structured analysis:",
    "<destination_analysis>",
    "{{",
    f"   'first_supporting_elements': ['Element 1', 'Element 2', ...]",
    f"   'first_candidate_destination': '{EmailDestination.choices_str()}'",
    f"   'first_reasoning': 'Brief explanation'",
    f"   'second_supporting_elements': ['Element 1', 'Element 2', ...]",
    f"   'second_candidate_destination': '{EmailDestination.choices_str()}'",
    f"   'second_reasoning': 'Brief explanation'",
    f"   'final_choice': '{EmailDestination.choices_str()}'",
    "}}",
    "</destination_analysis>",
    "",
    "## 2. Final Decisions",
    "Based on your analysis, provide the final decisions in these tags:",
    "",
    "<mark_as_read>",
    "true or false",
    "</mark_as_read>",
    "<destination>",
    f"{EmailDestination.choices_str()}",
    "</destination>",
]

prompt = Prompt(
    template=create_prompt_template(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
    ),
    input_tags=["sender", "subject", "email_summary"],
    output_tag_patterns=[
        TagPattern("read_status_analysis"),
        TagPattern("destination_analysis"),
        TagPattern("mark_as_read"),
        TagPattern("destination", parser=EmailDestination.parse),
    ],
)