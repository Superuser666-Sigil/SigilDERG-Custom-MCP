import React, { useEffect, useState } from "react";
import { fetchConfig } from "../../api/admin";

export const ConfigPage: React.FC = () => {
  const [config, setConfig] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    fetchConfig()
      .then((data) => {
        setConfig(data);
        setError(null);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div>Loading config...</div>;
  if (error) return <div className="text-red-400">Error: {error}</div>;

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-semibold">Configuration</h2>
      <pre className="bg-slate-900 p-4 rounded overflow-auto border border-slate-800 text-sm">
        {JSON.stringify(config, null, 2)}
      </pre>
    </div>
  );
};
