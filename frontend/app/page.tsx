"use client";

import { useEffect, useRef, useState } from "react";

import { Shell, useSession } from "@/components/Shell";
import { api, ApiError, type AgentAnswer, type ContextCard, type LeadSummary } from "@/lib/api";

interface Message {
  role: "rep" | "agent";
  text: string;
  answer?: AgentAnswer;
}

const STATUS_LABEL: Record<AgentAnswer["status"], string> = {
  answered: "Answered",
  escalated: "Sent to your team lead",
  needs_clarification: "Needs more detail",
  use_role_play: "Try a practice session",
};

function Workspace() {
  const session = useSession();
  const [leads, setLeads] = useState<LeadSummary[]>([]);
  const [leadId, setLeadId] = useState<string | null>(null);
  const [card, setCard] = useState<{ card: ContextCard; stale: boolean } | null>(null);
  const [cardError, setCardError] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api<LeadSummary[]>("/crm/leads", {}, session.token)
      .then(setLeads)
      .catch((failure: Error) => setError(failure.message));
  }, [session.token]);

  useEffect(() => {
    setCard(null);
    setCardError(null);
    if (!leadId) return;
    api<{ card: ContextCard; stale: boolean }>(`/crm/leads/${encodeURIComponent(leadId)}/card`, {}, session.token)
      .then(setCard)
      .catch((failure: Error) =>
        setCardError(failure instanceof ApiError && failure.status === 404 ? "No context card yet." : failure.message),
      );
  }, [leadId, session.token]);

  useEffect(() => bottom.current?.scrollIntoView({ behavior: "smooth" }), [messages]);

  async function ask(event: { preventDefault(): void }) {
    event.preventDefault();
    const text = question.trim();
    if (!text) return;
    setMessages((current) => [...current, { role: "rep", text }]);
    setQuestion("");
    setBusy(true);
    setError(null);
    try {
      const answer = await api<AgentAnswer>(
        "/agent/ask",
        { method: "POST", body: JSON.stringify({ question: text, lead_id: leadId }) },
        session.token,
      );
      setMessages((current) => [...current, { role: "agent", text: answer.text, answer }]);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "The agent could not answer");
    } finally {
      setBusy(false);
    }
  }

  const lead = leads.find((item) => item.id === leadId);

  return (
    <div className="workspace">
      <aside className="panel">
        <h2>{session.role === "rep" ? "My leads" : "All leads"}</h2>
        <button className={`lead ${leadId === null ? "selected" : ""}`} onClick={() => setLeadId(null)}>
          <strong>General questions</strong>
          <span className="muted">No client selected</span>
        </button>
        {leads.map((item) => (
          <button
            key={item.id}
            className={`lead ${leadId === item.id ? "selected" : ""}`}
            onClick={() => setLeadId(item.id)}
          >
            <strong>{item.display_name}</strong>
            <span className="muted">
              {item.stage.replace("_", " ")}
              {session.role !== "rep" && ` · ${item.assigned_rep}`}
            </span>
          </button>
        ))}
        {leads.length === 0 && <p className="muted small">No leads assigned.</p>}
      </aside>

      <section className="panel chat">
        <h2>Ask {lead ? `about ${lead.display_name}` : "the team lead"}</h2>
        <div className="messages">
          {messages.length === 0 && (
            <p className="muted">
              Ask about projects, prices, payment plans or unit availability. Answers cite their sources; anything
              uncertain goes to your team lead.
            </p>
          )}
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.role}`}>
              <p dir="auto">{message.text}</p>
              {message.answer && (
                <div className="meta">
                  <span className={`badge ${message.answer.status}`}>{STATUS_LABEL[message.answer.status]}</span>
                  {message.answer.citations.map((citation) => (
                    <span key={`${citation.kind}:${citation.reference}`} className="cite">
                      {citation.kind}: {citation.reference}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
          {busy && <p className="muted">Thinking…</p>}
          <div ref={bottom} />
        </div>
        {error && <p className="error">{error}</p>}
        <form className="ask" onSubmit={ask}>
          <textarea
            dir="auto"
            value={question}
            maxLength={2000}
            placeholder="e.g. What is the monthly installment on A-101 with 10% down over 8 years?"
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) ask(e);
            }}
          />
          <button disabled={busy || !question.trim()}>Ask</button>
        </form>
      </section>

      <aside className="panel cardpanel">
        <h2>Context card</h2>
        {!leadId && <p className="muted">Pick a lead to see what the client said, wants and was promised.</p>}
        {cardError && <p className="muted">{cardError}</p>}
        {card && <CardView card={card.card} stale={card.stale} />}
      </aside>
    </div>
  );
}

function Items({ title, items }: { title: string; items: { text: string }[] }) {
  if (items.length === 0) return null;
  return (
    <>
      <h3>{title}</h3>
      <ul>
        {items.map((item, index) => (
          <li key={index} dir="auto">
            {item.text}
          </li>
        ))}
      </ul>
    </>
  );
}

function CardView({ card, stale }: { card: ContextCard; stale: boolean }) {
  return (
    <div className="contextcard">
      <p>
        <strong>{card.display_name}</strong> <span className="badge">{card.stage.replace("_", " ")}</span>
      </p>
      {stale && <p className="warn small">New interactions since this card was built.</p>}
      {card.best_time_to_call && <p className="small">Best time to call: {card.best_time_to_call}</p>}
      <Items title="Stated needs" items={card.stated_needs} />
      <Items title="Closing drivers" items={card.closing_drivers} />
      {card.open_concerns.length > 0 && (
        <>
          <h3>Open concerns</h3>
          <ul>
            {card.open_concerns.map((concern, index) => (
              <li key={index} dir="auto">
                {concern.concern}
                {concern.suggested_answer && <div className="muted small">Suggested: {concern.suggested_answer}</div>}
              </li>
            ))}
          </ul>
        </>
      )}
      <Items title="Promises made" items={card.promises} />
      <Items title="Talking points" items={card.talking_points} />
      {card.matched_units.length > 0 && (
        <>
          <h3>Matching units</h3>
          <ul>
            {card.matched_units.map((unit) => (
              <li key={unit.code}>
                {unit.code} · {unit.project} · {unit.bedrooms} BR · EGP {Number(unit.price_egp).toLocaleString()}
              </li>
            ))}
          </ul>
        </>
      )}
      <p className="muted small">
        Built {new Date(card.generated_at).toLocaleString()}
        {card.dropped_claims > 0 && ` · ${card.dropped_claims} unsupported claims removed`}
      </p>
    </div>
  );
}

export default function Home() {
  return (
    <Shell>
      <Workspace />
    </Shell>
  );
}
