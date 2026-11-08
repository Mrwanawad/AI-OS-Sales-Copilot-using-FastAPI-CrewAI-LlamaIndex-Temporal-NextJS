"use client";

import { useCallback, useEffect, useState } from "react";

import { Shell, useSession } from "@/components/Shell";
import { api, leadsATeam, type Candidate, type Ticket } from "@/lib/api";

function Escalations() {
  const session = useSession();
  const lead = leadsATeam(session);
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      setTickets(await api<Ticket[]>(lead ? "/escalations" : "/escalations/mine", {}, session.token));
      if (lead) setCandidates(await api<Candidate[]>("/escalations/knowledge", {}, session.token));
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not load");
    }
  }, [lead, session.token]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function act(path: string, body: object) {
    setError(null);
    try {
      await api(path, { method: "POST", body: JSON.stringify(body) }, session.token);
      await refresh();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Action failed");
    }
  }

  return (
    <div className="stack">
      <div className="row">
        <h1>{lead ? "Open escalations" : "My questions"}</h1>
        <button className="ghost" onClick={refresh}>
          Refresh
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {tickets.length === 0 && <p className="muted">Nothing here.</p>}
      {tickets.map((ticket) =>
        lead ? (
          <TicketReview key={ticket.id} ticket={ticket} onAct={act} />
        ) : (
          <div key={ticket.id} className="card">
            <p dir="auto">
              <strong>{ticket.question}</strong>
            </p>
            <p className="small muted">
              {new Date(ticket.created_at).toLocaleString()} · <span className={`badge ${ticket.status}`}>{ticket.status}</span>
            </p>
            {ticket.answer && (
              <p dir="auto">
                {ticket.answer} <span className="muted small">— {ticket.answered_by}</span>
              </p>
            )}
          </div>
        ),
      )}

      {lead && (
        <>
          <h2>Answers proposed for shared knowledge</h2>
          <p className="muted small">A different person from the one who answered must approve before reps can find it.</p>
          {candidates.length === 0 && <p className="muted">No proposals waiting.</p>}
          {candidates.map((candidate) => (
            <div key={candidate.id} className="card">
              <p dir="auto">
                <strong>{candidate.question}</strong>
              </p>
              <p dir="auto">{candidate.answer}</p>
              <p className="small muted">
                {candidate.project} · proposed by {candidate.proposed_by}
              </p>
              <div className="row">
                <button onClick={() => act(`/escalations/knowledge/${candidate.id}/approve`, {})}>Approve</button>
                <button
                  className="ghost"
                  onClick={() => {
                    const note = prompt("Why reject?");
                    if (note) act(`/escalations/knowledge/${candidate.id}/reject`, { note });
                  }}
                >
                  Reject
                </button>
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  );
}

function TicketReview({ ticket, onAct }: { ticket: Ticket; onAct: (path: string, body: object) => Promise<void> }) {
  const [answer, setAnswer] = useState("");
  const [share, setShare] = useState(false);

  return (
    <div className="card">
      <p dir="auto">
        <strong>{ticket.question}</strong>
      </p>
      <p className="small muted">
        From {ticket.asked_by ?? "unknown"} · {ticket.project ?? "no project"}
        {ticket.lead_id && ` · lead ${ticket.lead_id}`} · {ticket.reason}
      </p>
      {ticket.context && (
        <details>
          <summary className="small">Context</summary>
          <pre className="context">{ticket.context}</pre>
        </details>
      )}
      <textarea dir="auto" value={answer} onChange={(e) => setAnswer(e.target.value)} placeholder="Answer the rep" />
      {!ticket.lead_id && ticket.project && (
        <label className="check small">
          <input type="checkbox" checked={share} onChange={(e) => setShare(e.target.checked)} />
          Propose this answer for {ticket.project} knowledge
        </label>
      )}
      <div className="row">
        <button
          disabled={!answer.trim()}
          onClick={() =>
            onAct(`/escalations/${ticket.id}/answer`, {
              answer,
              share_with_project: share ? ticket.project : null,
            })
          }
        >
          Send answer
        </button>
        <button
          className="ghost"
          onClick={() => {
            const note = prompt("Why dismiss?");
            if (note) onAct(`/escalations/${ticket.id}/dismiss`, { note });
          }}
        >
          Dismiss
        </button>
      </div>
    </div>
  );
}

export default function EscalationsPage() {
  return (
    <Shell>
      <Escalations />
    </Shell>
  );
}
