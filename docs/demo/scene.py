"""Demo scene for the README GIF.

Needs the local stack and web app running (see README, "Run it locally"):
  python scripts/dev_stack.py        # API on :8000; set DEMO_BASE / DEMO_API to use other ports
  cd frontend && npm run dev
and an LLM key in .env (or GEMINI_API_KEY with LLM_PROVIDER=gemini). The lead's context card
must exist (send a signed lead_opened webhook for DEMO-LEAD-1, see scripts/smoke_e2e.py).
Record:   python docs/demo/scene.py
Compose:  python ~/.claude/skills/demo-gif/scripts/compose.py docs/demo/build --out docs/demo.gif
All data is the invented demo set in demo_data/.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/.claude/skills/demo-gif/scripts"))
from director import Director, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASE = os.environ.get("DEMO_BASE", "http://localhost:3000")
API = os.environ.get("DEMO_API", "http://localhost:8000")
PASSWORD = os.environ.get("DEMO_PASSWORD") or (ROOT / ".devdata" / "demo_password").read_text().strip()


def sign_in(username: str) -> str:
    """A ready session, so the recording starts on the workspace instead of the login form."""
    request = urllib.request.Request(
        f"{API}/auth/login",
        json.dumps({"username": username, "password": PASSWORD}).encode(),
        {"content-type": "application/json"},
    )
    body = json.load(urllib.request.urlopen(request, timeout=30))
    session = {
        "token": body["access_token"],
        "username": body["username"],
        "displayName": body["display_name"],
        "role": body["role"],
        "expiresAt": body["expires_at"],
    }
    return json.dumps(session)


async def scene(d: Director):
    p = d.page
    await d._ctx.add_init_script(
        f"try{{if(!sessionStorage.getItem('ai-os-session'))"
        f"sessionStorage.setItem('ai-os-session', {json.dumps(sign_in('rep-a'))})}}catch(e){{}}"
    )
    await d.goto(BASE, settle=1.4)
    box = p.get_by_placeholder("e.g. What is the monthly installment")
    answers = p.locator(".message.agent")

    async def ask(text: str, index: int, label: str):
        await d.type(box, text)
        await d.press("Enter", pause=0.2)
        await d.fast(answers.nth(index).wait_for(timeout=90_000), label=label)

    await d.caption("Open a lead", step=1)
    await d.click(p.get_by_role("button", name="Synthetic Client One"))
    await p.get_by_text("Stated needs").wait_for()
    await d.focus(p.locator(".cardpanel"), zoom=1.35)
    await d.hold(1.4)
    await d.unfocus()

    await d.caption("Ask, get a cited answer", step=2)
    await d.click(box)
    await ask("Payment plan and delivery for Demo Gardens?", 0, "Searching documents")
    await d.focus(answers.nth(0), zoom=1.4)
    await d.arrow(p.locator(".cite").first, "Cited sources")
    await d.hold(1.6)
    await d.clear_arrow()
    await d.unfocus()

    await d.caption("Risky asks go to your team lead", step=3)
    await d.click(box)
    await ask("Can I promise this client 25% off to close today?", 1, "Checking policy")
    await d.hold(2.0)


run(
    scene,
    out=str(Path(__file__).with_name("build")),
    title="AI OS Sales Copilot",
    subtitle="Cited answers from your documents, with escalation to a human",
    url_label="localhost",
)
