import os
import base64
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
try:
    from twilio.rest import Client as TwilioClient
    _HAS_TWILIO = True
except ImportError:
    _HAS_TWILIO = False

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USER)
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "http://localhost:5173")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")


_THREAT_COLORS = {
    "safe": "#22c55e",
    "suspicious": "#f59e0b",
    "critical": "#ef4444",
}


def _build_alert_email_html(ctx):
    """Render a branded HTML alert email. Returns (html, plain_fallback)."""
    threat_label = (ctx.get("threat_label") or "suspicious").lower()
    threat_color = _THREAT_COLORS.get(threat_label, "#f59e0b")
    confidence = ctx.get("confidence", 0.0)
    confidence_pct = f"{confidence * 100:.0f}"
    source = ctx.get("source", "Unknown source")
    timestamp = ctx.get("timestamp") or datetime.utcnow().strftime("%d %b %Y, %H:%M UTC")
    scene_description = ctx.get("scene_description") or "No description available."
    reasoning = ctx.get("reasoning") or "No reasoning available."
    has_snapshot = bool(ctx.get("snapshot_b64"))
    snapshot_html = (
        '<tr><td style="padding:0 32px 24px 32px;">'
        '<div style="font-size:12px;color:#64748b;font-weight:600;letter-spacing:1px;margin-bottom:8px;">CAPTURED FRAME</div>'
        '<img src="cid:snapshot" alt="Threat snapshot" style="width:100%;max-width:536px;border-radius:8px;border:1px solid #e2e8f0;display:block;">'
        '</td></tr>'
        if has_snapshot else ''
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Falantir Security Alert</title>
</head>
<body style="margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background-color:#f1f5f9;">
<table cellpadding="0" cellspacing="0" border="0" width="100%" style="background-color:#f1f5f9;padding:24px 0;">
<tr><td align="center">
<table cellpadding="0" cellspacing="0" border="0" width="600" style="background-color:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.1);max-width:600px;">

<tr>
<td style="background-color:#0f172a;padding:24px 32px;color:#ffffff;">
<table width="100%" cellpadding="0" cellspacing="0">
<tr>
<td>
<div style="font-size:14px;font-weight:600;letter-spacing:2px;color:#10b981;">FALANTIR</div>
<div style="font-size:18px;font-weight:700;margin-top:4px;color:#ffffff;">Security Alert</div>
</td>
<td align="right">
<span style="display:inline-block;padding:6px 14px;background-color:{threat_color};color:#ffffff;border-radius:999px;font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;">{threat_label}</span>
</td>
</tr>
</table>
</td>
</tr>

<tr>
<td style="padding:32px 32px 8px 32px;">
<div style="font-size:12px;color:#64748b;font-weight:600;letter-spacing:1px;margin-bottom:8px;">DETECTED</div>
<div style="font-size:22px;font-weight:700;color:#0f172a;line-height:1.3;">{threat_label.title()} activity ({confidence_pct}% confidence)</div>
</td>
</tr>

<tr>
<td style="padding:0 32px;">
<table width="100%" cellpadding="0" cellspacing="0" style="margin-top:20px;border-top:1px solid #e2e8f0;">
<tr>
<td style="padding:12px 0;width:100px;color:#64748b;font-size:13px;">Source</td>
<td style="padding:12px 0;color:#0f172a;font-size:14px;font-weight:500;">{source}</td>
</tr>
<tr>
<td style="padding:12px 0;color:#64748b;font-size:13px;border-top:1px solid #e2e8f0;">Time</td>
<td style="padding:12px 0;color:#0f172a;font-size:14px;font-weight:500;border-top:1px solid #e2e8f0;">{timestamp}</td>
</tr>
</table>
</td>
</tr>

<tr>
<td style="padding:24px 32px 0 32px;">
<div style="font-size:12px;color:#64748b;font-weight:600;letter-spacing:1px;margin-bottom:8px;">SCENE DESCRIPTION</div>
<div style="font-size:14px;color:#334155;line-height:1.6;font-style:italic;">{scene_description}</div>
</td>
</tr>

<tr>
<td style="padding:24px 32px 0 32px;">
<div style="padding:16px;background-color:#fff7ed;border-left:4px solid {threat_color};border-radius:6px;">
<div style="font-size:11px;color:{threat_color};font-weight:700;letter-spacing:1px;margin-bottom:6px;">WHY FLAGGED</div>
<div style="font-size:14px;color:#334155;line-height:1.5;">{reasoning}</div>
</div>
</td>
</tr>

{snapshot_html}

<tr>
<td style="padding:24px 32px 32px 32px;">
<table width="100%" cellpadding="0" cellspacing="0">
<tr><td align="center">
<a href="{DASHBOARD_URL}" style="display:inline-block;padding:14px 32px;background-color:#0f172a;color:#ffffff;text-decoration:none;border-radius:8px;font-weight:600;font-size:14px;">View in Dashboard</a>
</td></tr>
</table>
</td>
</tr>

<tr>
<td style="background-color:#f8fafc;padding:20px 32px;border-top:1px solid #e2e8f0;">
<div style="font-size:11px;color:#94a3b8;text-align:center;line-height:1.5;">
This alert was automatically generated by Falantir Security System.<br>
You are receiving this because your account is configured to receive security notifications.
</div>
</td>
</tr>

</table>
</td></tr>
</table>
</body>
</html>"""

    plain = (
        f"FALANTIR SECURITY ALERT\n"
        f"=======================\n\n"
        f"Threat: {threat_label.upper()} ({confidence_pct}% confidence)\n"
        f"Source: {source}\n"
        f"Time:   {timestamp}\n\n"
        f"Scene Description:\n{scene_description}\n\n"
        f"Why Flagged:\n{reasoning}\n\n"
        f"View in dashboard: {DASHBOARD_URL}\n"
    )
    return html, plain


def send_email(to_email, subject, body, html=None, image_b64=None):
    """
    Send an email. If `html` is provided, sends a multipart/alternative
    message with both plain text and HTML parts. If `image_b64` is provided
    (base64-encoded JPEG), attaches it inline with Content-ID 'snapshot'
    so the HTML can reference it as <img src="cid:snapshot">.
    """
    if not SMTP_USER or not SMTP_PASS:
        print(f"EMAIL SKIPPED (No credentials): To {to_email}")
        return False

    try:
        if html:
            msg = MIMEMultipart("related")
            msg["From"] = FROM_EMAIL
            msg["To"] = to_email
            msg["Subject"] = subject
            alt = MIMEMultipart("alternative")
            alt.attach(MIMEText(body, "plain"))
            alt.attach(MIMEText(html, "html"))
            msg.attach(alt)
            if image_b64:
                try:
                    img_bytes = base64.b64decode(image_b64)
                    img = MIMEImage(img_bytes, _subtype="jpeg")
                    img.add_header("Content-ID", "<snapshot>")
                    img.add_header("Content-Disposition", "inline", filename="snapshot.jpg")
                    msg.attach(img)
                except Exception as e:
                    print(f"EMAIL: failed to attach inline image — {e}")
        else:
            msg = MIMEMultipart()
            msg["From"] = FROM_EMAIL
            msg["To"] = to_email
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

        # Port 465 = implicit SSL (Hostinger's recommended port); 587 = STARTTLS.
        if int(SMTP_PORT) == 465:
            server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=20)
        else:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20)
            server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
        server.quit()
        print(f"EMAIL SENT (SMTP {SMTP_SERVER}:{SMTP_PORT}): To {to_email}")
        return True
    except Exception as e:
        print(f"EMAIL ERROR: {e}")
        return False


def send_sms(to_phone, message):
    if not _HAS_TWILIO or not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        print(f"SMS SKIPPED (No credentials or twilio not installed): To {to_phone}")
        return False

    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_phone
        )
        print(f"SMS SENT: To {to_phone}")
        return True
    except Exception as e:
        print(f"SMS ERROR: {e}")
        return False


def make_call(to_phone, message):
    if not _HAS_TWILIO or not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        print(f"CALL SKIPPED (No credentials or twilio not installed): To {to_phone}")
        return False

    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        twiml = f'<Response><Say>{message}</Say></Response>'
        client.calls.create(
            twiml=twiml,
            from_=TWILIO_PHONE_NUMBER,
            to=to_phone
        )
        print(f"CALL INITIATED: To {to_phone}")
        return True
    except Exception as e:
        print(f"CALL ERROR: {e}")
        return False


def notify_all(user_email, user_phone, message, alert_context=None):
    """
    Notify user via all available channels.

    If `alert_context` is supplied, the email is rendered as a branded
    HTML alert with optional inline snapshot. The plain `message` is still
    used for SMS (which has no HTML support and a 160-character limit).
    """
    # Fallback recipients: on the live deployment the register form does not
    # collect a phone, so user records have an empty phone -> SMS/calls get
    # silently skipped. Set ALERT_TO_NUMBER (and optionally ALERT_TO_EMAIL) in
    # the environment to guarantee a recipient. On a Twilio trial the number
    # must be verified.
    user_phone = (user_phone or os.getenv("ALERT_TO_NUMBER", "")).strip()
    user_email = (user_email or os.getenv("ALERT_TO_EMAIL", "")).strip()

    results = {}
    if user_email:
        if alert_context:
            html, plain = _build_alert_email_html(alert_context)
            threat_label = (alert_context.get("threat_label") or "suspicious").upper()
            subject = f"Falantir Alert: {threat_label} activity detected"
            results['email'] = send_email(
                user_email,
                subject,
                plain,
                html=html,
                image_b64=alert_context.get("snapshot_b64"),
            )
        else:
            results['email'] = send_email(user_email, "Falantir Security Alert", message)
    if user_phone:
        results['sms'] = send_sms(user_phone, message)
        results['call'] = make_call(user_phone, message)
    return results
