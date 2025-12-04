import React, { useState } from "react";
import { rebuildVectorIndex } from "../../api/admin";

export const VectorPage: React.FC = () => {
  const [repo, setRepo] = useState("");
  const [model, setModel] = useState("default");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRebuild = async (force: boolean) => {
    if (!confirm(`Are you sure you want to rebuild vector index for ${repo || "ALL repos"}?`)) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await rebuildVectorIndex(repo || undefined, force, model);
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-semibold">Vector Index</h2>

      <div className="flex gap-4 items-center flex-wrap">
        <input
          type="text"
          placeholder="Repo name (optional)"
          className="bg-slate-800 border border-slate-700 p-2 rounded text-slate-100"
          value={repo}
          onChange={e => setRepo(e.target.value)}
        />
        <input
          type="text"
          placeholder="Model (default)"
          className="bg-slate-800 border border-slate-700 p-2 rounded text-slate-100"
          value={model}
          onChange={e => setModel(e.target.value)}
        />
        <button
          onClick={() => handleRebuild(false)}
          disabled={loading}
          className="px-4 py-2 bg-yellow-600 hover:bg-yellow-500 rounded disabled:opacity-50"
        >
          Update
        </button>
        <button
          onClick={() => handleRebuild(true)}
          disabled={loading}
          className="px-4 py-2 bg-red-600 hover:bg-red-500 rounded disabled:opacity-50"
        >
          Force Rebuild
        </button>
      </div>

      {loading && <div>Processing (this may take a while)...</div>}
      {error && <div className="text-red-400">Error: {error}</div>}
      {result && (
        <pre className="bg-slate-900 p-4 rounded overflow-auto border border-slate-800">
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
};
