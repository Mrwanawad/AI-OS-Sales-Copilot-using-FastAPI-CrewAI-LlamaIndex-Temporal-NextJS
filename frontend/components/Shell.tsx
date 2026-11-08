"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

import { leadsATeam, loadSession, logout, type Session } from "@/lib/api";

const SessionContext = createContext<Session | null>(null);

export function useSession(): Session {
  const session = useContext(SessionContext);
  if (!session) throw new Error("useSession outside a signed-in page");
  return session;
}

/** Wraps signed-in pages: redirects to /login without a session and draws the top bar. */
export function Shell({ children }: { children: ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [session, setSession] = useState<Session | null>(null);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    const current = loadSession();
    if (!current) router.replace(`/login?next=${encodeURIComponent(pathname)}`);
    setSession(current);
    setChecked(true);
  }, [pathname, router]);

  if (!checked || !session) return <div className="center muted">Loading…</div>;

  const links = [
    { href: "/", label: "Workspace" },
    { href: "/escalations", label: leadsATeam(session) ? "Escalations" : "My questions" },
    { href: "/documents", label: "Documents" },
  ];

  return (
    <SessionContext.Provider value={session}>
      <header className="topbar">
        <span className="brand">AI OS</span>
        <nav>
          {links.map((link) => (
            <Link key={link.href} href={link.href} className={pathname === link.href ? "active" : ""}>
              {link.label}
            </Link>
          ))}
        </nav>
        <span className="who">
          {session.displayName} · {session.role.replace("_", " ")}
        </span>
        <button
          className="ghost"
          onClick={async () => {
            await logout(session).catch(() => undefined);
            router.replace("/login");
          }}
        >
          Log out
        </button>
      </header>
      <main className="page">{children}</main>
    </SessionContext.Provider>
  );
}
