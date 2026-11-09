// Thin client for the AI OS API. The token lives in sessionStorage, so closing the tab logs out.

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export type Role = "rep" | "team_lead" | "manager";

export interface Session {
  token: string;
  username: string;
  displayName: string;
  role: Role;
  expiresAt: string;
}

export interface LeadSummary {
  id: string;
  display_name: string;
  stage: string;
  assigned_rep: string;
}

export interface SourcedItem {
  text: string;
  source_ids: string[];
}

export interface ContextCard {
  lead_id: string;
  display_name: string;
  stage: string;
  generated_at: string;
  latest_interaction_at: string | null;
  timeline: { interaction_id: string; occurred_at: string; kind: string; summary: string | null }[];
  stated_needs: SourcedItem[];
  closing_drivers: SourcedItem[];
  open_concerns: { concern: string; source_ids: string[]; suggested_answer: string | null }[];
  promises: SourcedItem[];
  talking_points: SourcedItem[];
  preferences: { max_budget_egp: string | null; bedrooms: number | null; project: string | null };
  matched_units: { code: string; project: string; price_egp: string; bedrooms: number }[];
  dropped_claims: number;
  best_time_to_call: string | null;
}

export interface AgentAnswer {
  text: string;
  status: "answered" | "escalated" | "needs_clarification" | "use_role_play";
  intent: string;
  citations: { kind: string; reference: string }[];
  confidence: number;
  ticket_id: string | null;
}

export interface Ticket {
  id: string;
  question: string;
  reason: string;
  created_at: string;
  status: "open" | "answered" | "dismissed";
  project: string | null;
  lead_id: string | null;
  asked_by: string | null;
  context: string;
  answer: string | null;
  answered_by: string | null;
}

export interface Candidate {
  id: string;
  ticket_id: string;
  question: string;
  answer: string;
  project: string;
  proposed_by: string;
  status: string;
  expires_at: string | null;
}

export interface RegisteredDocument {
  document_id: string;
  project: string;
  kind: string;
  latest_version: number;
  uploaded_by: string;
  uploaded_at: string;
}

export interface IngestionStatus {
  state: "running" | "completed" | "failed" | "unknown";
  chunks: number | null;
  error: string | null;
}

const KEY = "ai-os-session";

export function loadSession(): Session | null {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return null;
    const session = JSON.parse(raw) as Session;
    return new Date(session.expiresAt) > new Date() ? session : null;
  } catch {
    return null;
  }
}

export function saveSession(session: Session | null): void {
  try {
    if (session) sessionStorage.setItem(KEY, JSON.stringify(session));
    else sessionStorage.removeItem(KEY);
  } catch {
    // Storage can be blocked; the session then lasts for this page only.
  }
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function api<T>(path: string, init: RequestInit = {}, token?: string): Promise<T> {
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const response = await fetch(`${API_URL}${path}`, { ...init, headers });
  if (response.status === 204) return undefined as T;
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : response.statusText;
    if (response.status === 401 && token) saveSession(null);
    throw new ApiError(response.status, detail || `Request failed (${response.status})`);
  }
  return body as T;
}

export async function login(username: string, password: string): Promise<Session> {
  const body = await api<{
    access_token: string;
    username: string;
    display_name: string;
    role: Role;
    expires_at: string;
  }>("/auth/login", { method: "POST", body: JSON.stringify({ username, password }) });
  const session: Session = {
    token: body.access_token,
    username: body.username,
    displayName: body.display_name,
    role: body.role,
    expiresAt: body.expires_at,
  };
  saveSession(session);
  return session;
}

export async function logout(session: Session): Promise<void> {
  try {
    await api("/auth/logout", { method: "POST" }, session.token);
  } finally {
    saveSession(null);
  }
}

export function leadsATeam(session: Session | null): boolean {
  return session !== null && session.role !== "rep";
}
