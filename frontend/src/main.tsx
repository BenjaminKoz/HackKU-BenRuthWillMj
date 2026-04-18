import React, { useState } from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { Learning } from "./Learning";
import { Game } from "./Game";
import "./index.css";

function Root() {
  const [tab, setTab] = useState("translate");

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <nav style={{
        background: '#1a2236',
        padding: '12px 24px',
        display: 'flex',
        gap: '20px',
        justifyContent: 'center',
        borderBottom: '2px solid #7c5cff',
        position: 'sticky',
        top: 0,
        zIndex: 1000,
        boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
      }}>
        <button 
          onClick={() => setTab("translate")} 
          style={{ 
            background: tab === "translate" ? 'var(--accent)' : 'transparent',
            border: tab === "translate" ? 'none' : '1px solid #3d4a71',
            padding: '8px 20px',
            fontSize: '16px'
          }}
        >
          ASL Translator
        </button>
        <button 
          onClick={() => setTab("learning")} 
          style={{ 
            background: tab === "learning" ? 'var(--accent)' : 'transparent',
            border: tab === "learning" ? 'none' : '1px solid #3d4a71',
            padding: '8px 20px',
            fontSize: '16px'
          }}
        >
          Learning
        </button>
        <button 
          onClick={() => setTab("game")} 
          style={{ 
            background: tab === "game" ? 'var(--accent)' : 'transparent',
            border: tab === "game" ? 'none' : '1px solid #3d4a71',
            padding: '8px 20px',
            fontSize: '16px'
          }}
        >
          ASL Game
        </button>
      </nav>
      <main style={{ flex: 1 }}>
        {tab === "translate" ? <App /> : tab === "learning" ? <Learning /> : <Game />}
      </main>
    </div>
  );
}

const rootElement = document.getElementById("root");
if (rootElement) {
  ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
      <Root />
    </React.StrictMode>
  );
}
