'use client';

import { useState, useRef, useEffect } from 'react';

type ChatMessage = { role: 'user' | 'ai'; text: string; confidence?: number; stats?: any };
type Region = { box: number[]; label: string };

export default function Home() {
  const [imageA, setImageA] = useState<string | null>(null);
  const [metaA, setMetaA] = useState<any>(null);
  const [imageB, setImageB] = useState<string | null>(null);
  const [metaB, setMetaB] = useState<any>(null);
  const [sarImage, setSarImage] = useState<string | null>(null);
  const [metaSar, setMetaSar] = useState<any>(null);

  const [loading, setLoading] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'A' | 'B' | 'SAR'>('A');
  const [viewMode, setViewMode] = useState<'Original' | 'Evidence' | 'Overlay'>('Overlay');

  const [query, setQuery] = useState("");
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([{ role: 'ai', text: 'SatQuery AI initialized. Upload satellite imagery to begin.' }]);
  const [analyzing, setAnalyzing] = useState(false);
  
  const [regions, setRegions] = useState<Region[]>([]);
  const [evidenceType, setEvidenceType] = useState<string | null>(null);
  const [hoveredRegion, setHoveredRegion] = useState<number | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  
  // Day 12: Report State
  const [reportData, setReportData] = useState<any>(null);
  
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [metrics, setMetrics] = useState({ analysis: 'Awaiting query...', confidence: '--%', trace: ['System idle'] as string[] });

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [chatHistory]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>, slot: 'A' | 'B' | 'SAR') => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(slot);
    setRegions([]);
    setConnectionError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/upload', { method: 'POST', body: formData });
      if (!response.ok) throw new Error("Server responded with error status");
      const data = await response.json();

      if (slot === 'A') { setMetaA(data.metadata); setImageA(data.preview); setActiveTab('A'); }
      else if (slot === 'B') { setMetaB(data.metadata); setImageB(data.preview); setActiveTab('B'); }
      else if (slot === 'SAR') { setMetaSar(data.metadata); setSarImage(data.preview); setActiveTab('SAR'); }
    } catch (error) {
      console.error("Upload failed", error);
      setConnectionError("Backend server unreachable. Ensure FastAPI is running on port 8000.");
    } finally {
      setLoading(null);
    }
  };

  const handleAnalyze = async () => {
    if (!query.trim()) return;

    const newChat = [...chatHistory, { role: 'user' as const, text: query }];
    setChatHistory(newChat);
    setQuery("");
    setAnalyzing(true);
    setRegions([]);
    setEvidenceType(null);
    setHoveredRegion(null);
    setConnectionError(null);
    setReportData(null);
    setViewMode('Overlay'); 
    setMetrics(prev => ({ ...prev, trace: ['Agent Controller -> Processing task...'] }));

    try {
      const activeMetadata = { ...(metaA || {}), ...(metaB || {}), ...(metaSar || {}) };
      
      const response = await fetch('http://localhost:8000/api/vqa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          image_base64: imageA || "", 
          query: query, 
          has_sar: !!sarImage, 
          has_bitemporal: !!imageB,
          image_b_base64: imageB || "",
          sar_base64: sarImage || "",
          metadata: activeMetadata // Pass metadata for backend cleaning
        }),
      });

      if (!response.ok) throw new Error("Analysis failed on server");
      const data = await response.json();
      
      setChatHistory([...newChat, { role: 'ai', text: data.answer, confidence: data.confidence, stats: data.evidence.stats }]);

      if (data.evidence && data.evidence.regions) {
        setRegions(data.evidence.regions);
        setEvidenceType(data.evidence.type);
      }

      setReportData(data.report_data);

      setMetrics({
        analysis: data.task,
        confidence: `${data.confidence}%`,
        trace: data.trace || ['Execution trace unavailable']
      });

      if (data.task === "Bi-Temporal Change Detection" && imageB) setActiveTab('B');
      if (data.task === "Cross-Modal Optical-SAR Fusion" && sarImage) setActiveTab('SAR');

    } catch (error) {
      console.error("Analysis failed", error);
      setConnectionError("Failed to reach AI backend. Verify uvicorn server status.");
      setChatHistory([...newChat, { role: 'ai', text: "Connection error: FastAPI backend is not responding." }]);
    } finally { 
      setAnalyzing(false); 
    }
  };

  const downloadReport = () => {
    if (!reportData) return;
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `SatQuery_Analysis_Report_${new Date().getTime()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getActiveImage = () => {
    if (activeTab === 'A') return imageA;
    if (activeTab === 'B') return imageB;
    if (activeTab === 'SAR') return sarImage;
    return null;
  };

  const getActiveMeta = () => {
    if (activeTab === 'A') return metaA;
    if (activeTab === 'B') return metaB;
    if (activeTab === 'SAR') return metaSar;
    return null;
  };

  const aspect = getActiveMeta()?.width ? `${getActiveMeta().width}/${getActiveMeta().height}` : '16/9';

  const getBoxStyle = (label: string, evidenceType: string | null, viewMode: string, isHovered: boolean) => {
    let color = 'green';
    if (evidenceType === 'change_mask') color = 'orange';
    if (evidenceType === 'fusion_mask') {
      if (label.toLowerCase().includes('water')) color = 'blue';
      if (label.toLowerCase().includes('built')) color = 'purple';
    }

    const isEvidence = viewMode === 'Evidence';
    const styles: Record<string, any> = {
      'orange': { border: 'border-orange-500', bgOver: 'bg-orange-500/30', bgEv: 'bg-orange-500/60', label: 'bg-orange-600' },
      'blue': { border: 'border-blue-500', bgOver: 'bg-blue-500/30', bgEv: 'bg-blue-500/60', label: 'bg-blue-600' },
      'purple': { border: 'border-purple-500', bgOver: 'bg-purple-500/30', bgEv: 'bg-purple-500/60', label: 'bg-purple-600' },
      'green': { border: 'border-green-500', bgOver: 'bg-green-500/20', bgEv: 'bg-green-500/50', label: 'bg-green-600' }
    };
    const s = styles[color];
    const hoverEffects = isHovered ? 'border-white shadow-[0_0_15px_rgba(255,255,255,0.8)] z-50 scale-[1.01]' : `${s.border} z-10`;

    return {
      wrapper: `absolute border-2 pointer-events-none transition-all duration-300 ${hoverEffects} ${isEvidence ? s.bgEv : s.bgOver}`,
      label: `absolute -top-6 left-0 text-white text-[10px] font-bold px-1.5 py-0.5 rounded shadow-lg whitespace-nowrap ${s.label}`
    };
  };

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-200 p-4 gap-4 font-sans">
      <header className="flex justify-between items-center pb-2 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold text-white tracking-tight">SatQuery AI</h1>
          {connectionError && (
            <span className="text-xs bg-red-900/60 border border-red-700 text-red-300 px-2.5 py-0.5 rounded">
              {connectionError}
            </span>
          )}
        </div>
        <span className="text-xs bg-slate-800 px-2 py-1 rounded text-slate-400">SIH26167 Prototype</span>
      </header>

      <div className="flex flex-1 gap-4 overflow-hidden">
        <aside className="w-72 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col gap-4 overflow-y-auto">
          <h2 className="text-sm font-bold text-slate-400 tracking-wider">DATA INPUTS</h2>
          <div className="flex flex-col gap-1">
            <label className="text-xs font-semibold text-slate-300">Optical Image A (Base)</label>
            <label className="bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg py-3 text-xs flex items-center justify-center cursor-pointer transition-colors">
              {loading === 'A' ? "Processing..." : (imageA ? "✓ Image A Loaded" : "+ Upload Image A")}
              <input type="file" className="hidden" accept=".tif,.tiff,.png,.jpg,.jpeg" onChange={(e) => handleFileUpload(e, 'A')} />
            </label>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs font-semibold text-slate-300">Optical Image B (Change Det.)</label>
            <label className="bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg py-3 text-xs flex items-center justify-center cursor-pointer transition-colors">
              {loading === 'B' ? "Processing..." : (imageB ? "✓ Image B Loaded" : "+ Upload Image B")}
              <input type="file" className="hidden" accept=".tif,.tiff,.png,.jpg,.jpeg" onChange={(e) => handleFileUpload(e, 'B')} />
            </label>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs font-semibold text-slate-300">SAR Image (Radar Fusion)</label>
            <label className="bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg py-3 text-xs flex items-center justify-center cursor-pointer transition-colors">
              {loading === 'SAR' ? "Processing..." : (sarImage ? "✓ SAR Image Loaded" : "+ Upload SAR Data")}
              <input type="file" className="hidden" accept=".tif,.tiff,.png,.jpg,.jpeg" onChange={(e) => handleFileUpload(e, 'SAR')} />
            </label>
          </div>
        </aside>

        <main className="flex-1 bg-black border border-slate-800 rounded-xl flex flex-col relative overflow-hidden">
          <div className="absolute top-3 left-3 z-10 flex gap-2 bg-slate-900/80 p-1 rounded-lg border border-slate-800 backdrop-blur-sm">
            <button onClick={() => setActiveTab('A')} className={`px-3 py-1 text-xs rounded transition-colors ${activeTab === 'A' ? 'bg-blue-600 text-white font-bold' : 'text-slate-400 hover:text-white'}`}>Image A</button>
            <button onClick={() => setActiveTab('B')} disabled={!imageB} className={`px-3 py-1 text-xs rounded transition-colors ${activeTab === 'B' ? 'bg-blue-600 text-white font-bold' : 'text-slate-500 disabled:opacity-40'}`}>Image B</button>
            <button onClick={() => setActiveTab('SAR')} disabled={!sarImage} className={`px-3 py-1 text-xs rounded transition-colors ${activeTab === 'SAR' ? 'bg-blue-600 text-white font-bold' : 'text-slate-500 disabled:opacity-40'}`}>SAR View</button>
          </div>

          {regions.length > 0 && (
            <div className="absolute top-3 right-3 z-10 flex gap-1 bg-slate-900/80 p-1 rounded-lg border border-slate-800 backdrop-blur-sm">
              <button onClick={() => setViewMode('Original')} className={`px-3 py-1 text-xs rounded transition-colors ${viewMode === 'Original' ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-white'}`}>Original</button>
              <button onClick={() => setViewMode('Evidence')} className={`px-3 py-1 text-xs rounded transition-colors ${viewMode === 'Evidence' ? 'bg-blue-600 text-white font-bold' : 'text-slate-400 hover:text-white'}`}>Evidence</button>
              <button onClick={() => setViewMode('Overlay')} className={`px-3 py-1 text-xs rounded transition-colors ${viewMode === 'Overlay' ? 'bg-green-600 text-white font-bold' : 'text-slate-400 hover:text-white'}`}>Overlay</button>
            </div>
          )}

          <div className="flex-1 w-full h-full flex items-center justify-center p-4 bg-zinc-950 overflow-hidden">
            {getActiveImage() ? (
              <div className="relative inline-flex min-w-[60%] min-h-[60%] max-w-full max-h-full items-center justify-center" style={{ aspectRatio: aspect, height: '10000px', maxWidth: '100%', maxHeight: '100%' }}>
                <img src={getActiveImage()!} alt="Satellite View" className={`absolute inset-0 w-full h-full object-fill transition-opacity duration-300 ${viewMode === 'Evidence' ? 'opacity-15 grayscale' : 'opacity-100'}`} />
                {viewMode !== 'Original' && regions.map((region, idx) => {
                  const style = getBoxStyle(region.label, evidenceType, viewMode, hoveredRegion === idx);
                  return (
                    <div 
                      key={idx}
                      className={style.wrapper}
                      style={{ top: `${region.box[0]}%`, left: `${region.box[1]}%`, height: `${region.box[2] - region.box[0]}%`, width: `${region.box[3] - region.box[1]}%` }}
                    >
                      <span className={style.label}>{region.label}</span>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-slate-600 flex flex-col items-center">
                <p className="text-sm font-medium">Satellite Image Viewer</p>
              </div>
            )}
          </div>
        </main>

        <aside className="w-80 bg-slate-900 border border-slate-800 rounded-xl flex flex-col">
          <div className="p-4 border-b border-slate-800 flex justify-between items-center">
            <h2 className="text-sm font-bold text-slate-400 tracking-wider">AI ASSISTANT</h2>
            {reportData && (
              <button 
                onClick={downloadReport} 
                className="text-[10px] font-bold bg-green-600/80 hover:bg-green-500 text-white px-2 py-1 rounded transition-colors"
              >
                Download Report
              </button>
            )}
          </div>
          <div className="flex-1 p-4 overflow-y-auto flex flex-col gap-4">
            {chatHistory.map((msg, idx) => (
              <div key={idx} className={`p-3 rounded-lg text-sm w-11/12 ${msg.role === 'user' ? 'bg-blue-900/50 text-blue-100 self-end ml-auto border border-blue-800' : 'bg-slate-800 text-slate-300'}`}>
                <p>{msg.text}</p>
                {msg.stats && (
                  <div className="mt-3 bg-slate-950 p-2 rounded border border-slate-700 text-xs text-slate-300">
                    <p className="font-bold text-slate-400 mb-1 tracking-wider text-[10px]">REPORT SUMMARY</p>
                    {Object.entries(msg.stats).map(([key, val]) => (
                      <div key={key} className="flex justify-between py-0.5 border-b border-slate-800 last:border-0">
                        <span>{key}</span>
                        <span className={String(val).includes('%') ? (String(val).startsWith('+') ? 'text-orange-400 font-bold' : 'text-blue-400 font-bold') : 'text-slate-400'}>{String(val)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {analyzing && <div className="text-xs text-slate-500 animate-pulse">Running live AI inference...</div>}
            <div ref={chatEndRef} />
          </div>
          <div className="p-4 border-t border-slate-800 bg-slate-900 rounded-b-xl flex flex-col gap-2">
            <textarea 
              className="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-sm focus:outline-none focus:border-blue-500 resize-none"
              rows={3}
              placeholder='e.g., "What changed?" or "Where is the water?"'
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAnalyze(); } }}
            ></textarea>
            <button onClick={handleAnalyze} disabled={analyzing || (!imageA && !imageB && !sarImage)} className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white font-medium py-2 rounded-lg transition-colors">Analyze</button>
          </div>
        </aside>
      </div>

      <div className="h-44 flex gap-4">
        <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
          <h3 className="text-xs font-bold text-slate-400 tracking-wider mb-2 border-b border-slate-800 pb-1">ANALYSIS</h3>
          <div className="flex-1 flex items-center justify-center text-slate-200 text-sm font-medium text-center">{metrics.analysis}</div>
        </div>
        
        <div className="w-48 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
          <h3 className="text-xs font-bold text-slate-400 tracking-wider mb-2 border-b border-slate-800 pb-1">CONFIDENCE</h3>
          <div className="flex-1 flex items-center justify-center text-4xl font-light text-slate-200">{metrics.confidence}</div>
        </div>
        
        <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col overflow-hidden">
          <h3 className="text-xs font-bold text-slate-400 tracking-wider mb-2 border-b border-slate-800 pb-1">EVIDENCE</h3>
          <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
            {regions.length > 0 ? (
              <ul className="text-sm space-y-1 mt-1">
                {regions.map((reg, i) => (
                  <li 
                    key={i} 
                    onMouseEnter={() => setHoveredRegion(i)}
                    onMouseLeave={() => setHoveredRegion(null)}
                    className="cursor-pointer hover:bg-slate-800 text-slate-300 hover:text-white px-2 py-1.5 rounded flex justify-between items-center transition-colors border border-transparent hover:border-slate-700"
                  >
                    <span>{reg.label}</span>
                    <span className="text-slate-500 text-[10px] uppercase tracking-widest">Hover to View</span>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="flex items-center justify-center h-full text-slate-500 text-sm">No regions detected</div>
            )}
          </div>
        </div>
        
        <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col overflow-hidden">
          <h3 className="text-xs font-bold text-slate-400 tracking-wider mb-2 border-b border-slate-800 pb-1">ANALYSIS TRACE</h3>
          <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
            {Array.isArray(metrics.trace) ? (
              <ul className="text-xs space-y-2 mt-1">
                {metrics.trace.map((step, i) => (
                  <li key={i} className="flex gap-2 items-start text-slate-300">
                    <span className={step.includes('FAILED') ? "text-red-500 font-bold" : "text-green-500 font-bold"}>
                      {step.includes('FAILED') ? '✗' : '✓'}
                    </span>
                    <span className="leading-tight">{step}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="flex items-center justify-center h-full text-slate-500 text-xs">{metrics.trace}</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}