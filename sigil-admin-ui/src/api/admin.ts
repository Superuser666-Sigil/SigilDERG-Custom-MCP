import { AdminClient } from "./client";

export const adminClient = new AdminClient();

// Try to load API key from localStorage if available
const savedKey = localStorage.getItem("sigil_admin_key");
if (savedKey) {
  adminClient.setApiKey(savedKey);
}

export interface AdminStatus {
  admin: {
    host: string;
    port: number;
    enabled: boolean;
  };
  repos: Record<string, string>;
  index: {
    path: string;
    has_embeddings: boolean;
    embed_model: string | null;
  };
  watcher: {
    enabled: boolean;
    watching: string[];
  };
}

export async function fetchStatus(): Promise<AdminStatus> {
  return adminClient.get<AdminStatus>("/admin/status");
}

export async function rebuildIndex(repo?: string, force = true) {
  return adminClient.post("/admin/index/rebuild", { repo, force });
}

export async function fetchIndexStats(repo?: string) {
  const query = repo ? `?repo=${encodeURIComponent(repo)}` : "";
  return adminClient.get(`/admin/index/stats${query}`);
}

export async function rebuildVectorIndex(
  repo?: string,
  force = true,
  model = "default",
) {
  return adminClient.post("/admin/vector/rebuild", { repo, force, model });
}

export interface LogTail {
  path: string;
  lines: string[];
}

export async function fetchLogsTail(n = 200): Promise<LogTail> {
  return adminClient.get<LogTail>(`/admin/logs/tail?n=${n}`);
}

export async function fetchConfig(): Promise<unknown> {
  return adminClient.get("/admin/config");
}
