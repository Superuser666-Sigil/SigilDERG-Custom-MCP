import React, { useState } from "react";
import { StatusPage } from "./features/status/StatusPage";
import { IndexPage } from "./features/index/IndexPage";
import { VectorPage } from "./features/vectors/VectorPage";
import { LogsPage } from "./features/logs/LogsPage";
import { ConfigPage } from "./features/config/ConfigPage";
import { adminClient } from "./api/admin";

type Tab = "status" | "index" | "vectors" | "logs" | "config";

export const App: React.FC = () => {
  const [tab, setTab] = useState<Tab>("status");
  const [apiKey, setApiKey] = useState(localStorage.getItem("sigil_admin_key") || "");
  const [showKeyInput, setShowKeyInput] = useState(false);

  const handleKeySave = () => {
    localStorage.setItem("sigil_admin_key", apiKey);
    adminClient.setApiKey(apiKey);
    setShowKeyInput(false);
    // Reload to refresh data with new key
    window.location.reload();
  };

  return (
    <div className="min-h-screen flex bg-slate-950 text-slate-100 font-sans">
      <aside className="w-64 border-r border-slate-800 p-4 flex flex-col">
        <h1 className="text-xl font-bold mb-6 text-blue-400">Sigil Admin</h1>
        <nav className="space-y-1 text-sm flex-1">
          <NavButton active={tab === "status"} onClick={() => setTab("status")}>Status</NavButton>
          <NavButton active={tab === "index"} onClick={() => setTab("index")}>Index</NavButton>
          <NavButton active={tab === "vectors"} onClick={() => setTab("vectors")}>Vectors</NavButton>
          <NavButton active={tab === "logs"} onClick={() => setTab("logs")}>Logs</NavButton>
          <NavButton active={tab === "config"} onClick={() => setTab("config")}>Config</NavButton>
        </nav>

        <div className="mt-auto border-t border-slate-800 pt-4">
          <button
            onClick={() => setShowKeyInput(!showKeyInput)}
            className="text-xs text-slate-500 hover:text-slate-300 w-full text-left"
          >
            {showKeyInput ? "Hide API Key Settings" : "API Key Settings"}
          </button>

          {showKeyInput && (
            <div className="mt-2 space-y-2">
              <input
                type="password"
                value={apiKey}
                onChange={e => setApiKey(e.target.value)}
                placeholder="Enter Admin API Key"
                className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs"
              />
              <button
                onClick={handleKeySave}
                className="w-full bg-blue-700 hover:bg-blue-600 text-xs py-1 rounded"
              >
                Save & Reload
              </button>
            </div>
          )}
        </div>
      </aside>
      <main className="flex-1 p-6 overflow-auto">
        {tab === "status" && <StatusPage />}
        {tab === "index" && <IndexPage />}
        {tab === "vectors" && <VectorPage />}
        {tab === "logs" && <LogsPage />}
        {tab === "config" && <ConfigPage />}
      </main>
    </div>
  );
};

const NavButton: React.FC<{ active: boolean; onClick: () => void; children: React.ReactNode }> = ({ active, onClick, children }) => (
  <button
    onClick={onClick}
    className={`w-full text-left px-3 py-2 rounded transition-colors ${
      active
        ? "bg-slate-800 text-white font-medium"
        : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
    }`}
  >
    {children}
  </button>
);
