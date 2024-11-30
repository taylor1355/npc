from enum import Enum
from typing import Self
import logging

from npc.prompts.prompt_common import create_prompt_template, Prompt, TagPattern


class EmailDestination(Enum):
    NEWSLETTER = "newsletter"
    BUSINESS_TRANSACTION = "business_transaction"
    ARCHIVE = "archive"
    DELETE = "delete"
    KEEP_IN_INBOX = "keep_in_inbox"

    @classmethod
    def choices_str(cls) -> str:
        """Get a string containing all valid destination choices."""
        return "/".join(dest.name for dest in cls)

    def parse(text: str) -> Self:
        """Convert a string to an EmailDestination enum.

        Args:
            text: String representation of destination

        Returns:
            Corresponding EmailDestination enum value, defaults to KEEP_IN_INBOX if invalid
        """
        try:
            return EmailDestination[text.strip().upper()]
        except KeyError:
            logging.warning(f"Invalid destination value: {text}. Defaulting to KEEP_IN_INBOX.")
            return EmailDestination.KEEP_IN_INBOX


SYSTEM_PROMPT = [
    "You are an expert email organization assistant specializing in inbox management and workflow optimization."
    " Your task is to analyze an email and determine whether it requires user intervention and what its most"
    " appropriate destination should be."
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
    "## 1. User Intervention Assessment",
    "First, analyze whether the email requires user attention. Consider the following factors:",
    "",
    "Requires User Intervention:",
    "- Personal communications from friends, family, or important contacts",
    "- Time-sensitive notifications (e.g., payment deadlines, appointment reminders)",
    "- Security-related communications (2FA codes, account alerts)",
    "- Direct requests or questions requiring a response",
    "- Important business communications needing review or action",
    "",
    "Does Not Require User Intervention:",
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
    "- Regular content publications",
    "- Industry updates",
    "- Curated content digests",
    "",
    "BUSINESS_TRANSACTION:",
    "- Receipts and invoices",
    "- Payment confirmations",
    "- Account statements",
    "- Terms of service updates",
    "- Privacy policy changes",
    "",
    "ARCHIVE:",
    "- Completed transactions",
    "- Past event information",
    "- Reference materials",
    "- Resolved discussions",
    "",
    "DELETE:",
    "- Spam",
    "- Marketing material or promotions",
    "- Outdated announcements",
    "- Duplicate messages",
    "- Temporary notifications (e.g., one-time codes)",
    "",
    "KEEP_IN_INBOX:",
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
    "In <user_intervention_analysis> tags, provide a JSON-structured analysis:",
    "<user_intervention_analysis>",
    "{{",
    "   'intervention_required_reasons': 'Factors supporting user intervention'",
    "   'intervention_not_required_reasons': 'Factors supporting no user intervention'",
    "}}",
    "</user_intervention_analysis>",
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
    "<user_intervention_required>",
    "true or false",
    "</user_intervention_required>",
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
        TagPattern("user_intervention_analysis"),
        TagPattern("destination_analysis"),
        TagPattern("user_intervention_required"),
        TagPattern("destination", parser=EmailDestination.parse),
    ],
)

if __name__ == "__main__":
    print("\n".join(SYSTEM_PROMPT + USER_PROMPT))
