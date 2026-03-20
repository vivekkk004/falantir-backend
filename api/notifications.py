import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
try:
    from twilio.rest import Client as TwilioClient
    _HAS_TWILIO = True
except ImportError:
    _HAS_TWILIO = False

# Load settings from environment variables
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USER)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

def send_email(to_email, subject, body):
    if not SMTP_USER or not SMTP_PASS:
        print(f"EMAIL SKIPPED (No credentials): To {to_email}")
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = FROM_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
        server.quit()
        print(f"EMAIL SENT: To {to_email}")
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
        # Using a simple TWIML bin or voice URL. 
        # For a quick implementation, we can use a direct TWIML XML.
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

def notify_all(user_email, user_phone, message):
    """Notify user via all available channels."""
    results = {}
    if user_email:
        results['email'] = send_email(user_email, "ShopGuard Security Alert", message)
    if user_phone:
        results['sms'] = send_sms(user_phone, message)
        results['call'] = make_call(user_phone, message)
    return results
