# cron_jobs.py
import modal
from datetime import datetime, timedelta, timezone
from supabase import create_client
from dateutil import parser

from modal import (
    App,
    Image,
    Secret,
    asgi_app,
)

app_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "uvicorn",
        "fastapi",
        "pydantic",
        "requests",
        "supabase"
    )
)
# Initialize Modal App
app = App(
    "flara-jobs",
    image=app_image,
)


# Supabase client setup (replace with actual URL and key)
url = "https://jmlaffbdapwhgwovikel.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImptbGFmZmJkYXB3aGd3b3Zpa2VsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcxOTY5Njg1OSwiZXhwIjoyMDM1MjcyODU5fQ.YoFiCmxkgM4u9XFN1Su_qxXtNsv4MIgsx-Sh7UfxIXM"
supabase = create_client(url, key)

# Function to replenish daily credits
@app.function(schedule=modal.Cron("0 0 * * *"))  # Runs every 24 hours at midnight UTC
def replenish_daily_credits():
    users = supabase.table("user").select("user_id").execute()
    for user in users.data:
        supabase.table("user").update({"daily_credits": 400}).eq("user_id", user["user_id"]).execute()

# Function to reset organization credits for users
@app.function(schedule=modal.Cron("0 0 * * *"))  # Runs every 24 hours at midnight UTC
def reset_organization_credits():
    organizations = supabase.table("organizations").select("name", "daily_credits", "premium_credits").execute()
    users = supabase.table("user").select("user_id", "organization").execute()

    org_credits_map = {org["name"]: org["daily_credits"] + org["premium_credits"] for org in organizations.data}

    for user in users.data:
        org_name = user["organization"]
        if org_name in org_credits_map:
            supabase.table("user").update({"organization_credits": org_credits_map[org_name]}).eq("user_id", user["user_id"]).execute()

# Function to delete pending reports older than 30 minutes
@app.function(schedule=modal.Period(hours=1))  # Runs every hour
def delete_old_pending_reports():
    now = datetime.now(timezone.utc)
    # Check speech reports
    reports = supabase.table("speech_reports").select("id", "created_at", "pending").eq("pending", True).execute()

    for report in reports.data:
        #created_at = datetime.fromisoformat(report["created_at"].replace("Z", "+00:00"))
        created_at = parser.parse(report["created_at"])
        if (now - created_at) > timedelta(minutes=30):
            supabase.table("speech_reports").delete().eq("id", report["id"]).execute()

    # Check interview reports
    interviews = supabase.table("interview_reports").select("id", "created_at", "pending").eq("pending", True).execute()

    for interview in interviews.data:
        created_at = parser.parse(report["created_at"])
        if (now - created_at) > timedelta(minutes=30):
            supabase.table("interview_reports").delete().eq("id", interview["id"]).execute()

