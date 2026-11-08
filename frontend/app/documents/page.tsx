"use client";

import { useCallback, useEffect, useState, type FormEvent } from "react";

import { Shell, useSession } from "@/components/Shell";
import { api, leadsATeam, type IngestionStatus, type RegisteredDocument } from "@/lib/api";

const KINDS = ["brochure", "price_sheet", "payment_plan", "master_plan", "faq", "policy", "playbook"];
const ACCEPT = ".pdf,.docx,.pptx,.xlsx,.xls,.md,.html,.htm,.txt,.csv";

interface Tracked {
  documentId: string;
  version: number;
  statusUrl: string;
  status: IngestionStatus | null;
}

function Documents() {
  const session = useSession();
  const canUpload = leadsATeam(session);
  const [documents, setDocuments] = useState<RegisteredDocument[]>([]);
  const [tracked, setTracked] = useState<Tracked[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(() => {
    api<RegisteredDocument[]>("/knowledge/documents", {}, session.token)
      .then(setDocuments)
      .catch((failure: Error) => setError(failure.message));
  }, [session.token]);

  useEffect(refresh, [refresh]);

  // Polls running ingestions until each completes or fails.
  useEffect(() => {
    const running = tracked.filter((item) => !item.status || item.status.state === "running");
    if (running.length === 0) return;
    const timer = setTimeout(async () => {
      const updates = await Promise.all(
        running.map(async (item) => ({
          key: `${item.documentId}:${item.version}`,
          status: await api<IngestionStatus>(item.statusUrl, {}, session.token).catch(() => null),
        })),
      );
      setTracked((current) =>
        current.map((item) => {
          const update = updates.find((u) => u.key === `${item.documentId}:${item.version}`);
          return update?.status ? { ...item, status: update.status } : item;
        }),
      );
      if (updates.some((u) => u.status && u.status.state !== "running")) refresh();
    }, 2000);
    return () => clearTimeout(timer);
  }, [tracked, session.token, refresh]);

  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    setBusy(true);
    setError(null);
    try {
      const accepted = await api<{ document_id: string; version: number; status_url: string }>(
        "/knowledge/documents",
        { method: "POST", body: new FormData(form) },
        session.token,
      );
      setTracked((current) => [
        { documentId: accepted.document_id, version: accepted.version, statusUrl: accepted.status_url, status: null },
        ...current,
      ]);
      form.reset();
      refresh();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      <h1>Project documents</h1>
      {canUpload ? (
        <form className="card upload" onSubmit={upload}>
          <p className="muted small">
            Uploading a file under an existing document id replaces it: reps see the new version once ingestion finishes,
            and never a gap.
          </p>
          <label>
            Document id
            <input name="document_id" pattern="[a-z0-9][a-z0-9_\-]{1,79}" placeholder="demo-bay-brochure" required />
          </label>
          <label>
            Project
            <input name="project" placeholder="Demo Bay" required />
          </label>
          <label>
            Kind
            <select name="kind" defaultValue="brochure">
              {KINDS.map((kind) => (
                <option key={kind} value={kind}>
                  {kind.replace("_", " ")}
                </option>
              ))}
            </select>
          </label>
          <label>
            File
            <input name="file" type="file" accept={ACCEPT} required />
          </label>
          <button disabled={busy}>{busy ? "Uploading…" : "Upload"}</button>
        </form>
      ) : (
        <p className="muted">Team leads and managers add documents. Ask yours if something is missing.</p>
      )}
      {error && <p className="error">{error}</p>}

      {tracked.length > 0 && (
        <div className="card">
          <h2>This session&apos;s uploads</h2>
          <ul>
            {tracked.map((item) => (
              <li key={`${item.documentId}:${item.version}`}>
                {item.documentId} v{item.version}:{" "}
                <span className={`badge ${item.status?.state ?? "running"}`}>{item.status?.state ?? "queued"}</span>
                {item.status?.chunks != null && ` · ${item.status.chunks} chunks`}
                {item.status?.error && <span className="error small"> {item.status.error}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}

      <table>
        <thead>
          <tr>
            <th>Document</th>
            <th>Project</th>
            <th>Kind</th>
            <th>Version</th>
            <th>Uploaded</th>
          </tr>
        </thead>
        <tbody>
          {documents.map((document) => (
            <tr key={document.document_id}>
              <td>{document.document_id}</td>
              <td>{document.project}</td>
              <td>{document.kind.replace("_", " ")}</td>
              <td>{document.latest_version}</td>
              <td>
                {new Date(document.uploaded_at).toLocaleString()} · {document.uploaded_by}
              </td>
            </tr>
          ))}
          {documents.length === 0 && (
            <tr>
              <td colSpan={5} className="muted">
                No uploaded documents yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

export default function DocumentsPage() {
  return (
    <Shell>
      <Documents />
    </Shell>
  );
}
