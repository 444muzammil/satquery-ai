'use client';

import { useState, useRef, useEffect } from 'react';

type ChatMessage = { role: 'user' | 'ai'; text: string; confidence?: number; stats?: any };
type BoundingBox = number[];

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
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([{ role: 'ai', text: 'Welcome. Upload Image A and B, then ask "What changed?"' }]);
  const [analyzing, setAnalyzing] = useState(false);
  
  const [regions, setRegions] = useState<BoundingBox[]>([]);
  const [regionLabel, setRegionLabel] = useState<string>("");
  const [evidenceType, setEvidenceType] = useState<string | null>(null);
  
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [metrics, setMetrics] = useState({ analysis: 'Awaiting query...', confidence: '--%', evidence: 'No regions detected', trace: 'System idle' });

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [chatHistory]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>, slot: 'A' | 'B' | 'SAR') => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(slot);
    setRegions([]);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/upload', { method: 'POST', body: formData });
      const data = await response.json();

      if (slot === 'A') { setMetaA(data.metadata); setImageA(data.preview); setActiveTab('A'); }
      else if (slot === 'B') { setMetaB(data.metadata); setImageB(data.preview); setActiveTab('B'); }
      else if (slot === 'SAR') { setMetaSar(data.metadata); setSarImage(data.preview); setActiveTab('SAR'); }
    } catch (error) { console.error("Upload failed", error); } finally { setLoading(null); }
  };

  const handleAnalyze = async () => {
    if (!query.trim()) return;

    const newChat = [...chatHistory, { role: 'user' as const, text: query }];
    setChatHistory(newChat);
    setQuery("");
    setAnalyzing(true);
    setRegions([]);
    setEvidenceType(null);
    setViewMode('Overlay'); 
    setMetrics(prev => ({ ...prev, trace: 'Agent Controller -> Executing Task...' }));

    try {
      const response = await fetch('http://localhost:8000/api/vqa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: imageA || "", query: query, has_sar: !!sarImage, has_bitemporal: !!imageB }),
      });

      const data = await response.json();
      
      setChatHistory([...newChat, { role: 'ai', text: data.answer, confidence: data.confidence, stats: data.evidence.stats }]);

      if (data.evidence && data.evidence.coords) {
        setRegions(data.evidence.coords);
        setRegionLabel(data.evidence.label || "Detected");
        setEvidenceType(data.evidence.type);
      }

      setMetrics({
        analysis: data.task,
        confidence: `${data.confidence}%`,
        evidence: data.evidence.stats ? 'Statistical Change Report Generated' : (data.evidence.coords ? `${data.evidence.coords.length} region(s) mapped` : 'Text Analysis'),
        trace: data.trace || `Model Executed: ${data.model_used}`
      });

      if (data.task === "Bi-Temporal Change Detection" && imageB) setActiveTab('B');
      if (data.task === "Cross-Modal Optical-SAR Fusion" && sarImage) setActiveTab('SAR');

    } catch (error) {
      console.error("Analysis failed", error);
      setChatHistory([...newChat, { role: 'ai', text: "Error connecting to AI backend." }]);
    } finally { setAnalyzing(false); }
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

  const isChangeMap = evidenceType === 'change_mask';
  const boxBorderClass = isChangeMap ? 'border-orange-500' : 'border-green-500';
  const boxBgOverlay = isChangeMap ? 'bg-orange-500/30' : 'bg-green-500/20';
  const boxBgEvidence = isChangeMap ? 'bg-orange-500/60' : 'bg-blue-500/40';
  const labelBgClass = isChangeMap ? 'bg-orange-600' : (viewMode === 'Evidence' ? 'bg-blue-600' : 'bg-green-600');

  // Dynamically calculate aspect ratio to allow image scaling while keeping bounding boxes intact
  const activeMeta = getActiveMeta();
  const aspect = activeMeta && activeMeta.width && activeMeta.height ? `${activeMeta.width}/${activeMeta.height}` : '16/9';

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-200 p-4 gap-4 font-sans">
      <header className="flex justify-between items-center pb-2 border-b border-slate-800">
        <h1 className="text-2xl font-bold text-white tracking-tight">SatQuery AI</h1>
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
              // SCALING FIX: This perfectly forces the image to scale to the max size while keeping aspect ratio intact for the boxes
              <div 
                className="relative" 
                style={{ 
                  aspectRatio: aspect, 
                  height: '10000px', // Forces expansion; constrained by max-height
                  maxWidth: '100%', 
                  maxHeight: '100%' 
                }}
              >
                <img 
                  src={getActiveImage()!} 
                  alt="Satellite View" 
                  className={`absolute inset-0 w-full h-full object-fill transition-opacity duration-300 ${viewMode === 'Evidence' ? 'opacity-15 grayscale' : 'opacity-100'}`} 
                />
                
                {viewMode !== 'Original' && regions.map((bbox, idx) => (
                  <div 
                    key={idx}
                    className={`absolute border-2 pointer-events-none transition-all duration-500 ${viewMode === 'Evidence' ? `border-blue-500 ${boxBgEvidence}` : `${boxBorderClass} ${boxBgOverlay}`}`}
                    style={{ top: `${bbox[0]}%`, left: `${bbox[1]}%`, height: `${bbox[2] - bbox[0]}%`, width: `${bbox[3] - bbox[1]}%` }}
                  >
                    <span className={`absolute -top-6 left-0 text-white text-[10px] font-bold px-1.5 py-0.5 rounded shadow-lg whitespace-nowrap ${labelBgClass}`}>
                      {regionLabel} {idx + 1}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-slate-600 flex flex-col items-center">
                <p className="text-sm font-medium">Satellite Image Viewer</p>
              </div>
            )}
          </div>
        </main>

        <aside className="w-80 bg-slate-900 border border-slate-800 rounded-xl flex flex-col">
          <div className="p-4 border-b border-slate-800"><h2 className="text-sm font-bold text-slate-400 tracking-wider">AI ASSISTANT</h2></div>
          <div className="flex-1 p-4 overflow-y-auto flex flex-col gap-4">
            {chatHistory.map((msg, idx) => (
              <div key={idx} className={`p-3 rounded-lg text-sm w-11/12 ${msg.role === 'user' ? 'bg-blue-900/50 text-blue-100 self-end ml-auto border border-blue-800' : 'bg-slate-800 text-slate-300'}`}>
                <p>{msg.text}</p>
                
                {msg.stats && (
                  <div className="mt-3 bg-slate-950 p-2 rounded border border-slate-700 text-xs text-slate-300">
                    <p className="font-bold text-slate-400 mb-1 tracking-wider text-[10px]">CHANGE SUMMARY</p>
                    {Object.entries(msg.stats).map(([key, val]) => (
                      <div key={key} className="flex justify-between py-0.5 border-b border-slate-800 last:border-0">
                        <span>{key}</span>
                        <span className={String(val).startsWith('+') ? 'text-orange-400 font-bold' : 'text-green-400 font-bold'}>{String(val)}</span>
                      </div>
                    ))}
                  </div>
                )}
                
                {msg.confidence && <p className="text-[10px] text-slate-500 mt-2 pt-1 border-t border-slate-700">Confidence: {msg.confidence}%</p>}
              </div>
            ))}
            {analyzing && <div className="text-xs text-slate-500 animate-pulse">Agent is generating statistics...</div>}
            <div ref={chatEndRef} />
          </div>
          <div className="p-4 border-t border-slate-800 bg-slate-900 rounded-b-xl flex flex-col gap-2">
            <textarea 
              className="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-sm focus:outline-none focus:border-blue-500 resize-none"
              rows={3}
              placeholder='e.g., "What changed?"'
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAnalyze(); } }}
            ></textarea>
            <button onClick={handleAnalyze} disabled={analyzing || (!imageA && !imageB && !sarImage)} className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white font-medium py-2 rounded-lg transition-colors">Analyze</button>
          </div>
        </aside>
      </div>

      <div className="h-40 flex gap-4">
        <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col"><h3 className="text-xs font-bold text-slate-400 tracking-wider mb-2">ANALYSIS</h3><div className="flex-1 flex items-center justify-center text-slate-200 text-sm font-medium text-center">{metrics.analysis}</div></div>
        <div className="w-48 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col"><h3 className="text-xs font-bold text-slate-400 tracking-wider mb-2">CONFIDENCE</h3><div className="flex-1 flex items-center justify-center text-3xl font-light text-slate-200">{metrics.confidence}</div></div>
        <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col"><h3 className="text-xs font-bold text-slate-400 tracking-wider mb-2">EVIDENCE</h3><div className="flex-1 flex items-center justify-center text-slate-400 text-sm text-center px-2">{metrics.evidence}</div></div>
        <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col"><h3 className="text-xs font-bold text-slate-400 tracking-wider mb-2">EXECUTION TRACE</h3><div className="flex-1 flex items-center justify-center text-slate-400 text-[10px] text-center px-2 leading-relaxed">{metrics.trace}</div></div>
      </div>
    </div>
  );
}