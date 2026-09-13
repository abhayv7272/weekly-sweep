"""
emailer.py — send the weekly report over Gmail SMTP.

The HTML report is sent BOTH ways on purpose:
  * inline as the message body  -> you read it without opening anything
  * as a .html attachment       -> the pixel-perfect copy, opens in a browser,
                                   survives Gmail's message-size clipping

Credentials come from the environment (GitHub Actions secrets); nothing is
hard-coded.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr, formatdate
from typing import Dict, List, Optional, Sequence, Tuple

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

# Gmail clips a message body past ~102 kB. Charts live in the attachment anyway,
# so keep the inline body under that and let the attachment carry the full thing.
INLINE_LIMIT = 95_000


def send_report(subject: str, html_body: str, attachments: Sequence[Tuple[str, bytes, str]],
                to: Optional[str] = None, user: Optional[str] = None,
                password: Optional[str] = None, sender_name: str = "Weekly Sweep") -> None:
    """
    attachments: sequence of (filename, raw_bytes, mime_subtype) e.g.
                 ("report.html", b"...", "html")
    """
    # Secret names: MY_EMAIL / MY_APP_PASSWORD are what this repo uses; the
    # GMAIL_* names are accepted too so either convention works.
    user = user or os.environ.get("MY_EMAIL") or os.environ.get("GMAIL_USER") or ""
    password = (password
                or os.environ.get("MY_APP_PASSWORD")
                or os.environ.get("GMAIL_APP_PASSWORD")
                or "").replace(" ", "")
    # No separate recipient secret needed: default to sending it to yourself.
    to = to or os.environ.get("REPORT_TO") or user

    missing = [n for n, v in (("MY_EMAIL", user), ("MY_APP_PASSWORD", password)) if not v]
    if missing:
        raise RuntimeError(
            "Missing secret(s): " + ", ".join(missing) +
            ". Add them at Settings → Secrets and variables → Actions.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr((sender_name, user))
    msg["To"] = to
    msg["Date"] = formatdate(localtime=True)

    inline = html_body
    if len(inline.encode("utf-8")) > INLINE_LIMIT:
        # Body too big for Gmail: send a pointer, keep the full report attached.
        inline = _clipped_notice(html_body)

    msg.set_content(
        "This report is HTML. Your client is showing the plain-text fallback — "
        "open the attached .html file for the charts and tables.")
    msg.add_alternative(inline, subtype="html")

    for name, blob, subtype in attachments:
        maintype = "image" if subtype in ("png", "jpeg", "gif") else "application"
        if subtype in ("html", "csv", "plain"):
            maintype, sub = "text", subtype
        elif subtype == "png":
            maintype, sub = "image", "png"
        else:
            maintype, sub = "application", subtype
        msg.add_attachment(blob, maintype=maintype, subtype=sub, filename=name)

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx, timeout=120) as s:
        s.login(user, password)
        s.send_message(msg)
    print(f"· email sent to {to} ({len(attachments)} attachment(s))", flush=True)


def _clipped_notice(full_html: str) -> str:
    """Trim the body at a chart boundary so Gmail does not clip mid-report."""
    cut = full_html.rfind("<div style='background:#111827;border:1px solid #1f2a3a;"
                          "border-radius:10px;padding:15px;margin-bottom:14px'>",
                          0, INLINE_LIMIT)
    head = full_html[:cut] if cut > 1000 else full_html[:INLINE_LIMIT]
    return (head +
            "<div style=\"max-width:1080px;margin:0 auto;padding:18px;background:#111827;"
            "border:1px solid #1f2a3a;border-radius:10px;font:400 13px "
            "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#e5e7eb\">"
            "<b style='color:#38bdf8'>The remaining charts are in the attached "
            "report.html</b><br><span style='color:#94a3b8'>Gmail truncates long messages, "
            "so the rest of the gallery was moved to the attachment — open it for the "
            "complete, identical report.</span></div></div></div></body></html>")
