'use client';

import { useState } from 'react';

// Define types for our chat history
type ChatMessage = {
  role: 'user' | 'ai';
  text: string;
  confidence?: number;
};

export default function Home() {
  // Image Upload State
  const [imageA, setImageA] = useState<string | null>(null);
  const [metaA, setMetaA] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  // VQA Chat State
  const [query, setQuery] = useState("");
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([
    { role: 'ai', text: 'Welcome to SatQuery AI. Upload imagery to begin analysis.' }
  ]);
  const [analyzing, setAnalyzing] = useState(false);

  // Metrics State
  const [metrics, setMetrics] = useState({
    analysis: 'Awaiting query...',
    confidence: '--%',
    evidence: 'No regions detected',
    trace: 'System idle'
  });

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setMetaA(data.metadata);
      setImageA(data.preview);
    } catch (error) {
      console.error("Upload failed", error);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!query.trim()) return;
    
    // Add user query to chat
    const newChat = [...chatHistory, { role: 'user' as const, text: query }];
    setChatHistory(newChat);
    setQuery("");
    setAnalyzing(true);
    setMetrics(prev => ({ ...prev, trace: 'Agent Controller -> Validating Inputs...' }));

    try {
      const response = await fetch('http://localhost:8000/api/vqa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: imageA || "", // Send the base64 preview we got from upload
          query: query
        }),
      });
      
      const data = await response.json();
      
      // Add AI response to chat
      setChatHistory([...newChat, { role: 'ai', text: data.answer, confidence: data.confidence }]);
      
      // Update Metrics Dashboard
      setMetrics({
        analysis: 'Task Complete',
        confidence: `${data.confidence}%`,
        evidence: 'VLM Text Generation',
        trace: data.trace
      });

    } catch (error) {
      console.error("Analysis failed", error);
      setChatHistory([...newChat, { role: 'ai', text: "Error connecting to AI backend." }]);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-200 p-4 gap-4 font-sans">
      
      {/* HEADER */}
      <header className="flex justify-between items-center pb-2 border-b border-slate-800">
        <h1 className="text-2xl font-bold text-white tracking-tight">SatQuery AI</h1>
        <span className="text-xs bg-slate-800 px-2 py-1 rounded text-slate-400">SIH26167 Prototype</span>
      </header>

      {/* TOP SECTION: 3 Columns */}
      <div className="flex flex-1 gap-4 overflow-hidden">
        
        {/* LEFT PANEL: DATA */}
        <aside className="w-72 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col gap-6 overflow-y-auto">
          <h2 className="text-sm font-bold text-slate-400 tracking-wider">DATA</h2>
          
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">Image A (Optical 1)</label>
            <label className="bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg py-6 text-sm flex flex-col items-center justify-center transition-colors cursor-pointer">
              {loading ? "Processing..." : "+ Upload File"}
              <input type="file" className="hidden" accept=".tif,.tiff,.png,.jpg,.jpeg" onChange={handleFileUpload} />
            </label>
            
            {metaA && (
              <div className="bg-slate-950 p-3 rounded border border-slate-800 text-xs text-slate-400 space-y-1 mt-2">
                <p className="font-bold text-slate-200 mb-2">IMAGE INFORMATION</p>
                <p>Format: {metaA.format}</p>
                <p>Dimensions: {metaA.width} × {metaA.height}</p>
                <p>Bands: {metaA.bands}</p>
                <p>CRS: {metaA.crs}</p>
                {metaA.resolution !== "N/A" && <p>Resolution: {metaA.resolution[0]?.toFixed(2)}</p>}
              </div>
            )}
          </div>

          <div className="flex flex-col gap-2 opacity-50 pointer-events-none">
            <label className="text-sm">Image B (Optical 2)</label>
            <button className="bg-slate-800 border border-slate-700 rounded-lg py-6 text-sm flex items-center justify-center">
              + Upload
            </button>
          </div>
          <div className="flex flex-col gap-2 opacity-50 pointer-events-none">
            <label className="text-sm">SAR Image</label>
            <button className="bg-slate-800 border border-slate-700 rounded-lg py-6 text-sm flex items-center justify-center">
              + Upload
            </button>
          </div>
        </aside>

        {/* MIDDLE PANEL: SATELLITE VIEWER */}
        <main className="flex-1 bg-black border border-slate-800 rounded-xl flex items-center justify-center relative overflow-hidden">
          {imageA ? (
            <img src={imageA} alt="Satellite Preview" className="object-contain w-full h-full" />
          ) : (
            <div className="text-slate-600 flex flex-col items-center">
              <svg className="w-12 h-12 mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-sm font-medium">Satellite Image Viewer</p>
            </div>
          )}
        </main>

        {/* RIGHT PANEL: AI CHAT */}
        <aside className="w-80 bg-slate-900 border border-slate-800 rounded-xl flex flex-col">
          <div className="p-4 border-b border-slate-800">
            <h2 className="text-sm font-bold text-slate-400 tracking-wider">AI ASSISTANT</h2>
          </div>
          
          <div className="flex-1 p-4 overflow-y-auto flex flex-col gap-4">
            {chatHistory.map((msg, idx) => (
              <div key={idx} className={`p-3 rounded-lg text-sm w-11/12 ${msg.role === 'user' ? 'bg-blue-900/50 text-blue-100 self-end ml-auto border border-blue-800' : 'bg-slate-800 text-slate-300'}`}>
                <p>{msg.text}</p>
                {msg.confidence && (
                  <p className="text-xs text-slate-500 mt-2 border-t border-slate-700 pt-1">
                    Confidence: {msg.confidence}%
                  </p>
                )}
              </div>
            ))}
            {analyzing && (
               <div className="text-xs text-slate-500 animate-pulse">Agent is analyzing...</div>
            )}
          </div>

          <div className="p-4 border-t border-slate-800 bg-slate-900 rounded-b-xl flex flex-col gap-2">
            <textarea 
              className="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-sm focus:outline-none focus:border-blue-500 resize-none"
              rows={3}
              placeholder="Ask about your imagery..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAnalyze(); } }}
            ></textarea>
            <button 
              onClick={handleAnalyze}
              disabled={analyzing || !imageA}
              className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-medium py-2 rounded-lg transition-colors"
            >
              Analyze
            </button>
          </div>
        </aside>

      </div>

      {/* BOTTOM SECTION: METRICS */}
      <div className="h-40 flex gap-4">
        <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
          <h3 className="text-xs font-bold text-slate-400 tracking-wider mb-2">ANALYSIS</h3>
          <div className="flex-1 flex items-center justify-center text-slate-400 text-sm font-medium">{metrics.analysis}</div>
        </div>
        <div className="w-48 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
          <h3 className="text-xs font-bold text-slate-400 tracking-wider mb-2">CONFIDENCE</h3>
          <div className="flex-1 flex items-center justify-center text-3xl font-light text-slate-400">{metrics.confidence}</div>
        </div>
        <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
          <h3 className="text-xs font-bold text-slate-400 tracking-wider mb-2">EVIDENCE</h3>
          <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">{metrics.evidence}</div>
        </div>
        <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
          <h3 className="text-xs font-bold text-slate-400 tracking-wider mb-2">EXECUTION TRACE</h3>
          <div className="flex-1 flex items-center justify-center text-slate-500 text-sm text-center px-2">{metrics.trace}</div>
        </div>
      </div>

    </div>
  );
}