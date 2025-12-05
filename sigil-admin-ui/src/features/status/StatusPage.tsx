import React, { useEffect, useState } from "react";
import { AdminStatus, fetchStatus } from "../../api/admin";

export const StatusPage: React.FC = () => {
  const [status, setStatus] = useState<AdminStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetchStatus()
      .then((data) => {
        setStatus(data);
        setError(null);
      })
      .catch((err) => {
        setError(err.message || String(err));
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading && !status) {
    return <div>Loading status...</div>;
  }

  if (error) {
    return <div className="text-red-400">Error: {error}</div>;
  }

  if (!status) {
    return <div>No status data.</div>;
  }

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-semibold">Server Status</h2>

      <section className="bg-slate-900 border border-slate-800 p-4 rounded">
        <h3 className="font-semibold mb-2">Admin API</h3>
        <p>Host: <code>{status.admin.host}</code></p>
        <p>Port: {status.admin.port}</p>
        <p>Enabled: {String(status.admin.enabled)}</p>
      </section>

      <section className="bg-slate-900 border border-slate-800 p-4 rounded">
        <h3 className="font-semibold mb-2">Repositories</h3>
        <ul className="text-sm space-y-1">
          {Object.entries(status.repos).map(([name, path]) => (
            <li key={name}>
              <span className="font-mono">{name}</span>{" "}
              <span className="text-slate-400">→ {path}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="bg-slate-900 border border-slate-800 p-4 rounded">
        <h3 className="font-semibold mb-2">Index</h3>
        <p>Path: <code>{status.index.path}</code></p>
        <p>Embeddings configured: {String(status.index.has_embeddings)}</p>
        <p>Embedding model: {status.index.embed_model ?? "none"}</p>
      </section>

      <section className="bg-slate-900 border border-slate-800 p-4 rounded">
        <h3 className="font-semibold mb-2">Watcher</h3>
        <p>Enabled: {String(status.watcher.enabled)}</p>
        {status.watcher.enabled && (
          <p>
            Watching repos:{" "}
            {status.watcher.watching.length
              ? status.watcher.watching.join(", ")
              : "none"}
          </p>
        )}
      </section>
    </div>
  );
};
