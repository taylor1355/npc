import logging
from enum import Enum
from typing import Self

from npc.prompts.prompt_common import create_prompt_template, Prompt, TagPattern


class EmailPriority(Enum):
    URGENT = "URGENT"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    def parse(text: str) -> Self:
        """Convert a string to an EmailPriority enum.

        Args:
            text: String representation of priority level

        Returns:
            Corresponding EmailPriority enum value, defaults to MEDIUM if invalid
        """
        try:
            return EmailPriority[text.strip().upper()]
        except KeyError:
            logging.warning(f"Invalid priority value: {text}. Defaulting to MEDIUM.")
            return EmailPriority.MEDIUM


SYSTEM_PROMPT = [
    "You are an expert email analyst tasked with extracting key information from emails and providing structured summaries."
    "Your goal is to help users quickly understand and prioritize their incoming messages."
]

# TODO: email_summarizer needs to handle new input fields (received_datetime, current_date). received_datetime should be to the minute.
# TODO: email_summarizer needs to handle new output fields (priority_analysis, link_i, link_description_i)
# TODO: email_summarizer needs to invalidate the cache for any emails from previous days. record the last cache update time.
USER_PROMPT = [
    "# Email Details",
    "Here are the details of the email you need to analyze:",
    "<subject>{subject}</subject>",
    "<sender>{sender}</sender>",
    "<received_datetime>{received_datetime}</received_datetime>",
    "<current_date>{current_date}</current_date>",
    "<body>{body}</body>",
    "",
    "# Instructions",
    "Please follow these steps to analyze the email:",
    "",
    "## 1. Email Type",
    "Identify the email type from the following list and output it in <type> tags.",
    "",
    "Email Types:",
    "- Action Required",
    "- Scheduled Events",
    "- Personal Correspondence",
    "- Newsletters",
    "- Marketing Material",
    "- Notifications and Alerts",
    "- Social Updates",
    "",
    "## 2. Priority",
    "Determine the email's priority level. Use the following guidelines:",
    "URGENT:",
    "- Deadlines within 1-2 days",
    "- Late notices or payment reminders",
    "- Non-routine government communications",
    "- Urgent messages from friends/family",
    "- Account security alerts",
    "- Time-sensitive opportunities",
    "",
    "HIGH:",
    "- Deadlines within 1-2 weeks",
        "- Bills or payments due soon without autopay setup",
    "- Important event RSVPs",
    "- Medical/appointment scheduling",
    "- Job/education communications",
    "- Personal requests from close contacts",
    "",
    "MEDIUM:",
    "- Deadlines beyond 2 weeks",
    "- Non-urgent personal correspondence",
    "- Subscription renewals",
    "",
    "LOW:",
    "- Promotional/marketing emails",
    "- Newsletters",
    "- Social media notifications",
    "- Routine account updates",
    "- Informational updates (no action needed)",
    "  - Routine shipping notifications",
    "  - Receipts or confirmations",
    "  - Scheduled payment reminders",
    "  - etc.",
    "",
    "In <priority_analysis> tags, consider at least two potential priority levels for the email."
    " For each priority level, list specific elements from the email that support that level."
    " Then explain your reasoning for each, and make a final decision."
    " Structure your analysis as a JSON object, following the template below.",
    "- Make sure to be an advocate for the user. Be skeptical, and remain aware that emails often want to grab the user's attention.",
    "<priority_analysis>",
    "{{",
    "   'first_supporting_elements': ['Element 1', 'Element 2', ...]",
    "   'first_candidate_priority': 'URGENT/HIGH/MEDIUM/LOW'",
    "   'first_reasoning': 'Explanation'",
    "   'second_supporting_elements': ['Element 1', 'Element 2', ...]",
    "   'second_candidate_priority': 'URGENT/HIGH/MEDIUM/LOW'",
    "   'second_reasoning': 'Explanation'",
    "   'final_choice': 'URGENT/HIGH/MEDIUM/LOW'",
    "}}",
    "</priority_analysis>",
    "",
    "After your analysis, state the final priority level in <priority> tags in the following format.",
    "<priority>URGENT/HIGH/MEDIUM/LOW</priority>",
    "",
    "## 3. Link Extraction (for Medium or Higher Priority)",
    "If the email's priority is Medium or higher, extract up to three important links from the email body. For each link:",
    "- Place the URL in <link_i> tags, where i is the link number (link_1, link_2, link_3).",
    "- Provide a brief description of where the link leads in <link_description_i> tags.",
    "",
    "## 4. Summary",
    "Finally, provide a clear, concise summary in <summary></summary> tags. Include:",
    "- The main content or purpose of the email",
    "- Any required actions and their deadlines",
    "- Why this email is noteworthy or why it can be deprioritized",
    "- For medium or higher priority emails, mention the presence of important links",
]

prompt = Prompt(
    template=create_prompt_template(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
    ),
    input_tags=["subject", "sender", "received_datetime", "current_date", "body"],
    output_tag_patterns=[
        TagPattern("type"),
        TagPattern("priority_analysis"),
        TagPattern("priority", parser=EmailPriority.parse),
        TagPattern("link_\\d+", name="links", templated=True),
        TagPattern("link_description_\\d+", name="link_descriptions", templated=True),
        TagPattern("summary"),
    ],
)
