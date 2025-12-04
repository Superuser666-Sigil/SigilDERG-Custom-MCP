import React, { useState } from "react";
import { fetchIndexStats, rebuildIndex } from "../../api/admin";

export const IndexPage: React.FC = () => {
  const [repo, setRepo] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);

  const handleStats = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await fetchIndexStats(repo || undefined);
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRebuild = async (force: boolean) => {
    if (!confirm(`Are you sure you want to rebuild index for ${repo || "ALL repos"}?`)) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await rebuildIndex(repo || undefined, force);
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-semibold">Index Management</h2>

      <div className="flex gap-4 items-center">
        <input
          type="text"
          placeholder="Repo name (optional)"
          className="bg-slate-800 border border-slate-700 p-2 rounded text-slate-100"
          value={repo}
          onChange={e => setRepo(e.target.value)}
        />
        <button
          onClick={handleStats}
          disabled={loading}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded disabled:opacity-50"
        >
          Get Stats
        </button>
        <button
          onClick={() => handleRebuild(false)}
          disabled={loading}
          className="px-4 py-2 bg-yellow-600 hover:bg-yellow-500 rounded disabled:opacity-50"
        >
          Update Index
        </button>
        <button
          onClick={() => handleRebuild(true)}
          disabled={loading}
          className="px-4 py-2 bg-red-600 hover:bg-red-500 rounded disabled:opacity-50"
        >
          Force Rebuild
        </button>
      </div>

      {loading && <div>Loading...</div>}
      {error && <div className="text-red-400">Error: {error}</div>}
      {result && (
        <pre className="bg-slate-900 p-4 rounded overflow-auto border border-slate-800">
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
};
