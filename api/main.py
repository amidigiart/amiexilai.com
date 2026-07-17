# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import os
import time

import stripe
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from adapters import OpenAICompatAdapter
from crisis import detect_crisis, crisis_response
from dual_engine import DualEngine

GROK_URL = os.getenv("GROK_API_URL", "https://api.x.ai/v1")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4.5")
GROK_KEY = os.getenv("GROK_API_KEY", "")

DEEPSEEK_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")

MOCK_MODE = os.getenv("AMIEXILAI_MOCK", "false").lower() == "true"

STRIPE_SECRET = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
SITE_URL = os.getenv("SITE_URL", "https://amiexilai.com")

stripe.api_key = STRIPE_SECRET

ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS", "https://amiexilai.com,http://amiexilai.com,http://localhost:8000"
).split(",")

SYSTEM_PROMPT = (
    "You are amiExilAI — a warm, knowledgeable, and empathetic AI companion for "
    "people living abroad, especially Romanian diaspora families. You are an AI system "
    "(not a human) and say so if asked. You speak in the user's language "
    "(detect from their message — Romanian, English, German, Spanish, Italian, French).\n\n"
    "YOUR EXPERTISE:\n"
    "- Legal & administrative: residence permits, work visas, EU free movement rights, "
    "family reunification, consular services, apostille/legalization of documents, "
    "driving license conversion, voting from abroad\n"
    "- Taxes & finance: double taxation treaties, tax residency rules, sending money "
    "home, opening bank accounts abroad, pension transfers, social security coordination "
    "(EU Reg. 883/2004), child benefits cross-border\n"
    "- Healthcare: EHIC/CEAM, healthcare system navigation by country, registering with "
    "a GP, emergency procedures, prescription transfers, mental health resources\n"
    "- Education: school enrollment for children abroad, diploma recognition (ENIC-NARIC), "
    "university equivalences, bilingual education, maintaining mother tongue\n"
    "- Cultural integration: language learning strategies, understanding local customs, "
    "dealing with homesickness, building community abroad, cultural shock management\n"
    "- Practical relocation: housing (renting/buying), utilities setup, insurance types, "
    "postal services, mobile/internet, public transport systems\n"
    "- Return to Romania: re-integration steps, property buying, business opening, "
    "children's school transfer, pension/benefits transfer back\n\n"
    "OUTPUT FORMAT:\n"
    "- Use 🌍/📋/💰/🏥/📚/🏠/❤️ icons for visual clarity\n"
    "- Structure advice as step-by-step guides when appropriate\n"
    "- Always specify which country the advice applies to\n"
    "- Mention official sources and institutions by name\n\n"
    "CRITICAL RULES:\n"
    "- You are NOT a lawyer, tax advisor, or immigration consultant.\n"
    "- NEVER give definitive legal opinions — always recommend consulting a licensed "
    "professional for specific cases.\n"
    "- Laws and procedures change frequently — always advise verifying with the "
    "relevant official institution.\n"
    "- Be sensitive to the emotional dimension of expatriation — loneliness, family "
    "separation, identity questions are real and valid.\n"
    "- When discussing children's rights, always prioritize the child's best interest.\n"
    "- Do not help with illegal immigration or document fraud — full stop.\n"
    "- For emergencies abroad, always provide: local emergency number, Romanian "
    "consulate contact, and 112 (EU-wide)."
)

RATE_LIMIT: dict[str, list[float]] = {}


def _rate_ok(ip: str, max_req: int = 20, window: float = 60.0) -> bool:
    now = time.time()
    RATE_LIMIT[ip] = [t for t in RATE_LIMIT.get(ip, []) if now - t < window] + [now]
    return len(RATE_LIMIT[ip]) <= max_req


def _sign(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _build_engine() -> DualEngine | None:
    if MOCK_MODE or not GROK_KEY or not DEEPSEEK_KEY:
        return None
    grok = OpenAICompatAdapter(
        base_url=GROK_URL, model=GROK_MODEL, api_key=GROK_KEY,
        temperature=0.3, max_tokens=600, timeout=30, name="grok",
    )
    deepseek = OpenAICompatAdapter(
        base_url=DEEPSEEK_URL, model=DEEPSEEK_MODEL, api_key=DEEPSEEK_KEY,
        temperature=0.3, max_tokens=600, timeout=30, name="deepseek",
    )
    return DualEngine(grok, deepseek, system_prompt=SYSTEM_PROMPT, threshold=0.45)


ENGINE = _build_engine()

app = FastAPI(title="amiExilAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    locale: str = "RO"


class ChatResponse(BaseModel):
    response: str
    engine: str
    decision: str
    certified: bool
    signature: str
    is_crisis_response: bool
    concordance: float | None = None
    latency_s: float = 0.0


@app.get("/")
def root():
    return {"service": "amiExilAI API", "version": "1.0.0"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": "dual (Grok + DeepSeek)" if ENGINE else "mock",
        "grok_configured": bool(GROK_KEY),
        "deepseek_configured": bool(DEEPSEEK_KEY),
        "stripe_configured": bool(STRIPE_SECRET and STRIPE_PRICE_ID),
        "mock_mode": MOCK_MODE or not ENGINE,
    }


@app.post("/companion/exil/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    if not _rate_ok(ip):
        raise HTTPException(429, "Too many requests — please wait a moment")

    msg = req.message.strip()
    if not msg:
        raise HTTPException(400, "Empty message")
    if len(msg) > 2000:
        raise HTTPException(400, "Message too long (max 2000 chars)")

    locale = req.locale.upper()[:2] if req.locale else "RO"

    crisis = detect_crisis(msg)
    if crisis.is_crisis:
        resp = crisis_response(locale.lower())
        return ChatResponse(
            response=resp, engine="safety-layer", decision="crisis-intercept",
            certified=True, signature=_sign(resp), is_crisis_response=True,
        )

    if ENGINE:
        result = ENGINE.ask(msg)
        return ChatResponse(
            response=result.reply, engine=result.engine, decision=result.decision,
            certified=True, signature=_sign(result.reply), is_crisis_response=False,
            concordance=result.concordance, latency_s=result.latency_s,
        )

    mock = "Întrebare bună! În producție, verific cu două motoare AI independente. Revino curând pentru răspunsuri verificate."
    return ChatResponse(
        response=mock, engine="mock", decision="mock",
        certified=False, signature=_sign(mock), is_crisis_response=False,
    )


@app.post("/create-checkout-session")
def create_checkout():
    if not STRIPE_SECRET or not STRIPE_PRICE_ID:
        raise HTTPException(503, "Payment system not configured yet")
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url=SITE_URL + "?payment=success",
            cancel_url=SITE_URL + "?payment=cancelled",
        )
    except stripe.error.StripeError as e:
        raise HTTPException(502, f"Payment error: {e.user_message or str(e)}")
    return {"url": session.url}


@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(400, "Invalid webhook signature")
    if event["type"] == "checkout.session.completed":
        print(f"[STRIPE] New sub: {event['data']['object'].get('customer_email', '?')}")
    elif event["type"] == "customer.subscription.deleted":
        print(f"[STRIPE] Sub cancelled: {event['data']['object'].get('id')}")
    return {"received": True}


@app.post("/gdpr/data-request")
def gdpr_data():
    return {
        "message": "amiExilAI nu stochează mesaje sau conversații. Procesarea e stateless. "
                   "Pentru date Stripe, scrie la privacy@amiexilai.com.",
        "data_stored": "none",
    }


@app.delete("/gdpr/delete")
def gdpr_delete():
    return {
        "message": "Nicio dată personală nu e stocată pe server. "
                   "Pentru datele Stripe, scrie la privacy@amiexilai.com.",
        "status": "no_data_held",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
