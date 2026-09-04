"use client";

import React, { useState, useRef, useEffect } from 'react';
import { 
  Satellite, Crosshair, Upload, Map, Layers, Target, Activity, 
  CheckCircle2, AlertTriangle, ShieldCheck, Download, 
  ChevronRight, Database, Maximize, Cpu
} from 'lucide-react';

// =====================================================================
// TYPES & INTERFACES (Mapped to your exact backend structure)
// =====================================================================
interface Region {
  box: [number, number, number, number]; // [ymin_pct, xmin_pct, ymax_pct, xmax_pct]
  label: string;
  actual_pct: number;
}

interface AnalysisResult {
  answer: string;
  confidence: number;
  task: string;
  model_used: string;
  evidence: {
    type: string;
    regions: Region[];
    stats: Record<string, any>;
  };
  trace: string[];
  report_data?: any;
  gis_export?: any;
}

interface ImageState {
  file: File | null;
  preview: string;
  metadata: any;
  status: 'awaiting' | 'loading' | 'loaded' | 'error';
}

const defaultImageState: ImageState = { file: null, preview: "", metadata: {}, status: 'awaiting' };

// =====================================================================
// MAIN COMPONENT
// =====================================================================
export default function SatQueryApp() {
  // --- Data Input States ---
  const [imageA, setImageA] = useState<ImageState>(defaultImageState);
  const [imageB, setImageB] = useState<ImageState>(defaultImageState);
  const [imageSAR, setImageSAR] = useState<ImageState>(defaultImageState);

  // --- Viewer State ---
  const [activeTab, setActiveTab] = useState<'A' | 'B' | 'SAR'>('A');

  // --- Analysis States ---
  const [query, setQuery] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // --- Refs ---
  const fileInputA = useRef<HTMLInputElement>(null);
  const fileInputB = useRef<HTMLInputElement>(null);
  const fileInputSAR = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll trace to bottom when updating
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [result]);

  // =====================================================================
  // API INTEGRATION (Preserved exactly as requested)
  // =====================================================================
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>, type: 'A' | 'B' | 'SAR') => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    
    const setFn = type === 'A' ? setImageA : type === 'B' ? setImageB : setImageSAR;
    setFn({ file, preview: "", metadata: {}, status: 'loading' });

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/api/upload", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Upload failed");
      const data = await res.json();
      
      setFn({
        file,
        preview: data.preview,
        metadata: data.metadata,
        status: 'loaded'
      });
      setActiveTab(type); // Switch view to uploaded image
      setResult(null); // Clear previous results when new data is added
    } catch (err) {
      console.error(err);
      setFn({ file: null, preview: "", metadata: {}, status: 'error' });
    }
  };

  const handleAnalyze = async () => {
    if (!imageA.preview) {
      setError("Base optical imagery (Image A) is required.");
      return;
    }
    if (!query.trim()) {
      setError("Please enter an analysis query.");
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    const payload = {
      image_base64: imageA.preview,
      query: query,
      has_sar: !!imageSAR.preview,
      has_bitemporal: !!imageB.preview,
      image_b_base64: imageB.preview || "",
      sar_base64: imageSAR.preview || "",
      metadata: imageA.metadata || {},
      metadata_b: imageB.metadata || {},
      metadata_sar: imageSAR.metadata || {}
    };

    try {
      const res = await fetch("http://localhost:8000/api/vqa", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("Analysis request failed");
      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "An error occurred during analysis.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleDownloadGIS = () => {
    if (!result?.gis_export) return;
    const blob = new Blob([JSON.stringify(result.gis_export, null, 2)], { type: "application/geo+json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `SIH26167_Export_${Date.now()}.geojson`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // =====================================================================
  // HELPER COMPONENTS
  // =====================================================================
  const renderTraceLog = (traceString: string, idx: number) => {
    const match = traceString.match(/^\[(.*?)\]:\s*(.*)$/);
    if (!match) return <div key={idx} className="text-slate-400 font-mono text-xs py-1">{traceString}</div>;
    
    const tag = match[1];
    const content = match[2];
    
    let tagColor = "text-[#38BDF8]"; // Cyan default
    if (tag === "OBSERVATION") tagColor = "text-[#34D399]"; // Success green
    if (tag === "COMPATIBILITY") tagColor = "text-slate-400";
    if (content.includes("FAILED")) tagColor = "text-[#FB7185]"; // Critical red

    return (
      <div key={idx} className="font-mono text-xs mb-3 border-l-2 border-[#1C2A3A] pl-3 py-0.5">
        <div className={`font-semibold tracking-wide ${tagColor}`}>{tag.replace(/_/g, ' ')}</div>
        <div className="text-slate-300 mt-1 leading-relaxed">{content}</div>
      </div>
    );
  };

  // =====================================================================
  // MAIN UI RENDER
  // =====================================================================
  return (
    <div className="min-h-screen bg-[#070B12] text-[#F1F5F9] font-sans flex flex-col selection:bg-[#38BDF8] selection:text-[#070B12]">
      
      {/* HEADER */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-[#1C2A3A] bg-[#0D1420] shrink-0">
        <div className="flex items-center gap-3">
          <Satellite className="w-5 h-5 text-[#38BDF8]" />
          <div>
            <h1 className="text-sm font-bold tracking-widest text-[#F1F5F9]">SATQUERY AI</h1>
            <p className="text-[10px] uppercase tracking-widest text-[#64748B]">Remote Sensing Intelligence</p>
          </div>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-2 text-[#34D399]">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#34D399] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-[#34D399]"></span>
            </span>
            SYSTEM READY
          </div>
          <span className="px-2 py-1 rounded bg-[#111A27] border border-[#1C2A3A] text-[#94A3B8]">
            SIH26167
          </span>
        </div>
      </header>

      {/* MAIN GRID */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4 p-4 min-h-0">
        
        {/* LEFT PANEL: DATA INPUTS (col-span-3) */}
        <aside className="lg:col-span-3 flex flex-col gap-4">
          <div className="bg-[#0D1420] border border-[#1C2A3A] rounded-md flex flex-col h-full overflow-y-auto">
            <div className="p-3 border-b border-[#1C2A3A] flex items-center gap-2 bg-[#111A27] rounded-t-md">
              <Database className="w-4 h-4 text-[#94A3B8]" />
              <h2 className="text-xs font-semibold tracking-widest text-[#94A3B8] uppercase">Data Inputs</h2>
            </div>
            
            <div className="p-4 flex flex-col gap-4">
              {/* Input Card Generator */}
              {[
                { title: "OPTICAL / BASE", label: "Image A", state: imageA, ref: fileInputA, type: 'A' },
                { title: "BI-TEMPORAL / CHANGE", label: "Image B", state: imageB, ref: fileInputB, type: 'B' },
                { title: "RADAR / FUSION", label: "SAR Data", state: imageSAR, ref: fileInputSAR, type: 'SAR' }
              ].map((inp) => (
                <div key={inp.type} className="border border-[#1C2A3A] rounded bg-[#111A27] p-3 transition-colors hover:border-[#38BDF8]/30">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-[10px] font-mono tracking-widest text-[#38BDF8] flex items-center gap-1">
                      <span className="w-1 h-1 bg-[#38BDF8] rounded-full"></span> {inp.title}
                    </span>
                    {inp.state.status === 'loaded' && <CheckCircle2 className="w-3.5 h-3.5 text-[#34D399]" />}
                  </div>
                  
                  <div className="mb-3 text-sm text-[#F1F5F9] font-medium">{inp.label}</div>
                  
                  {inp.state.status === 'loaded' ? (
                    <div className="space-y-2">
                      <div className="text-xs text-[#94A3B8] font-mono truncate bg-[#070B12] p-1.5 rounded border border-[#1C2A3A]">
                        {inp.state.metadata.filename || "satellite_scene.tif"}
                      </div>
                      <div className="flex gap-2 text-[10px] text-[#64748B] font-mono">
                        <span>{inp.state.metadata.width}×{inp.state.metadata.height}</span>
                        <span>•</span>
                        <span>{inp.state.metadata.bands} BANDS</span>
                        <span>•</span>
                        <span>{inp.state.metadata.crs || "EPSG:4326"}</span>
                      </div>
                      <button 
                        onClick={() => inp.ref.current?.click()}
                        className="mt-2 w-full text-xs py-1.5 border border-[#1C2A3A] text-[#94A3B8] hover:text-[#F1F5F9] hover:bg-[#1C2A3A] transition-colors rounded"
                      >
                        Replace Dataset
                      </button>
                    </div>
                  ) : inp.state.status === 'loading' ? (
                    <div className="text-xs font-mono text-[#38BDF8] animate-pulse flex items-center gap-2">
                      <Activity className="w-3 h-3" /> Ingesting spatial data...
                    </div>
                  ) : (
                    <button 
                      onClick={() => inp.ref.current?.click()}
                      className="w-full border border-dashed border-[#1C2A3A] hover:border-[#38BDF8]/50 bg-[#070B12] hover:bg-[#0D1420] text-[#94A3B8] text-xs py-4 rounded transition-all flex flex-col items-center gap-2"
                    >
                      <Upload className="w-4 h-4" />
                      <span>Upload {inp.label}</span>
                    </button>
                  )}
                  
                  <input 
                    type="file" 
                    className="hidden" 
                    ref={inp.ref as React.RefObject<HTMLInputElement>} 
                    onChange={(e) => handleUpload(e, inp.type as 'A' | 'B' | 'SAR')} 
                    accept=".tif,.tiff,.png,.jpg,.jpeg" 
                  />
                </div>
              ))}
            </div>
          </div>
        </aside>

        {/* CENTER PANEL: SATELLITE VIEWER (col-span-6) */}
        <section className="lg:col-span-6 flex flex-col bg-[#0D1420] border border-[#1C2A3A] rounded-md overflow-hidden relative">
          
          {/* Tab Bar */}
          <div className="flex bg-[#111A27] border-b border-[#1C2A3A]">
            {[
              { id: 'A', label: 'IMAGE A', active: !!imageA.preview },
              { id: 'B', label: 'IMAGE B', active: !!imageB.preview },
              { id: 'SAR', label: 'SAR VIEW', active: !!imageSAR.preview }
            ].map((tab) => (
              <button
                key={tab.id}
                disabled={!tab.active}
                onClick={() => setActiveTab(tab.id as 'A' | 'B' | 'SAR')}
                className={`flex-1 py-2.5 text-xs font-mono tracking-widest border-r border-[#1C2A3A] last:border-r-0 transition-colors
                  ${activeTab === tab.id 
                    ? 'bg-[#0D1420] text-[#38BDF8] border-b-2 border-b-[#38BDF8]' 
                    : tab.active 
                      ? 'text-[#94A3B8] hover:bg-[#1C2A3A] hover:text-[#F1F5F9]' 
                      : 'text-[#64748B] opacity-50 cursor-not-allowed'
                  }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Viewer Area */}
          <div className="flex-1 relative bg-[#070B12] overflow-hidden flex items-center justify-center">
            
            {/* Empty State */}
            {(!imageA.preview && !imageB.preview && !imageSAR.preview) && (
              <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none p-6 text-center">
                <div className="absolute inset-0 opacity-5" style={{ backgroundImage: 'radial-gradient(#38BDF8 1px, transparent 1px)', backgroundSize: '24px 24px' }}></div>
                <Crosshair className="w-12 h-12 text-[#1C2A3A] mb-6" />
                <h3 className="text-[#F1F5F9] font-medium tracking-wide mb-2">SATQUERY AI</h3>
                <p className="text-[#94A3B8] text-sm mb-6 max-w-xs">Remote Sensing Workspace</p>
                <div className="px-4 py-2 border border-[#1C2A3A] rounded bg-[#0D1420] text-[#64748B] text-xs font-mono">
                  Select a dataset from Data Inputs to begin analysis
                </div>
                <p className="text-[#64748B] text-[10px] font-mono mt-4">GeoTIFF • TIFF • PNG • JPEG</p>
              </div>
            )}

            {/* Active Image Display */}
            {((activeTab === 'A' && imageA.preview) || 
              (activeTab === 'B' && imageB.preview) || 
              (activeTab === 'SAR' && imageSAR.preview)) && (
              <div className="relative w-full h-full p-4 flex items-center justify-center group">
                {/* Image Container with precise relative positioning for bounding boxes */}
                <div className="relative max-w-full max-h-full inline-block border border-[#1C2A3A] shadow-2xl">
                  <img 
                    src={activeTab === 'A' ? imageA.preview : activeTab === 'B' ? imageB.preview : imageSAR.preview} 
                    alt="Satellite Observation" 
                    className="max-w-full max-h-full object-contain"
                  />
                  
                  {/* Regions / Evidence Overlays */}
                  {result?.evidence?.regions?.map((reg, idx) => {
                    const [ymin, xmin, ymax, xmax] = reg.box;
                    const isWater = reg.label.toLowerCase().includes("water") || reg.label.toLowerCase().includes("flood");
                    const borderColor = isWater ? "border-[#38BDF8]" : "border-[#34D399]";
                    const bgColor = isWater ? "bg-[#38BDF8]/10" : "bg-[#34D399]/10";
                    
                    return (
                      <div 
                        key={idx}
                        className={`absolute border-2 ${borderColor} ${bgColor} pointer-events-none transition-all duration-500`}
                        style={{
                          top: `${ymin}%`,
                          left: `${xmin}%`,
                          height: `${ymax - ymin}%`,
                          width: `${xmax - xmin}%`
                        }}
                      >
                        <div className={`absolute -top-5 left-[-2px] px-1.5 py-0.5 text-[8px] font-mono whitespace-nowrap text-[#070B12] 
                          ${isWater ? 'bg-[#38BDF8]' : 'bg-[#34D399]'}`}>
                          {reg.label.toUpperCase()}
                        </div>
                      </div>
                    );
                  })}
                </div>
                
                {/* Subtle View Controls */}
                <div className="absolute top-4 right-4 flex flex-col gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button className="p-2 bg-[#0D1420]/80 border border-[#1C2A3A] rounded text-[#94A3B8] hover:text-[#F1F5F9] backdrop-blur"><Maximize className="w-4 h-4" /></button>
                  <button className="p-2 bg-[#0D1420]/80 border border-[#1C2A3A] rounded text-[#94A3B8] hover:text-[#F1F5F9] backdrop-blur"><Layers className="w-4 h-4" /></button>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* RIGHT PANEL: AI COPILOT (col-span-3) */}
        <aside className="lg:col-span-3 flex flex-col gap-4">
          <div className="bg-[#0D1420] border border-[#1C2A3A] rounded-md flex flex-col h-full overflow-hidden">
            <div className="p-3 border-b border-[#1C2A3A] flex items-center justify-between bg-[#111A27]">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-[#38BDF8]" />
                <h2 className="text-xs font-semibold tracking-widest text-[#94A3B8] uppercase">AI Copilot</h2>
              </div>
              <ShieldCheck className="w-4 h-4 text-[#34D399]" />
            </div>

            <div className="p-4 flex-1 flex flex-col justify-center">
              
              {!imageA.preview ? (
                <div className="text-center pb-8">
                  <div className="w-10 h-10 rounded-full bg-[#111A27] border border-[#1C2A3A] flex items-center justify-center mx-auto mb-4">
                    <Target className="w-4 h-4 text-[#64748B]" />
                  </div>
                  <p className="text-[#F1F5F9] text-sm mb-2">System Ready</p>
                  <p className="text-xs text-[#64748B]">Load base imagery to initiate copilot.</p>
                </div>
              ) : (
                <div className="flex flex-col h-full justify-end">
                  
                  {/* Suggestion Chips */}
                  <div className="mb-4 space-y-2">
                    <p className="text-[10px] font-mono text-[#94A3B8] uppercase tracking-wider mb-3">Suggested Tasks</p>
                    <button onClick={() => setQuery("Describe the land-cover and major objects visible in this image.")} className="w-full text-left px-3 py-2 text-xs bg-[#111A27] hover:bg-[#1C2A3A] border border-[#1C2A3A] rounded text-[#94A3B8] hover:text-[#F1F5F9] transition-colors truncate">
                      Describe major objects
                    </button>
                    {imageB.preview && (
                      <button onClick={() => setQuery("What changed between these two dates, and where did the change occur?")} className="w-full text-left px-3 py-2 text-xs bg-[#111A27] hover:bg-[#1C2A3A] border border-[#1C2A3A] rounded text-[#38BDF8]/80 hover:text-[#38BDF8] transition-colors truncate">
                        Analyze bi-temporal change
                      </button>
                    )}
                    {imageSAR.preview && (
                      <button onClick={() => setQuery("Use the optical and SAR images together to identify built-up and water-covered regions.")} className="w-full text-left px-3 py-2 text-xs bg-[#111A27] hover:bg-[#1C2A3A] border border-[#1C2A3A] rounded text-[#34D399]/80 hover:text-[#34D399] transition-colors truncate">
                        Extract cross-modal features
                      </button>
                    )}
                  </div>

                  {/* Query Input */}
                  <div className="relative mt-4">
                    <textarea 
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Ask about your imagery..."
                      className="w-full bg-[#070B12] border border-[#1C2A3A] rounded-md p-3 pb-12 text-sm text-[#F1F5F9] placeholder:text-[#64748B] focus:outline-none focus:border-[#38BDF8] transition-colors resize-none h-32"
                      disabled={isAnalyzing}
                    />
                    <button 
                      onClick={handleAnalyze}
                      disabled={isAnalyzing || !query.trim()}
                      className={`absolute bottom-3 right-3 px-4 py-1.5 text-xs font-semibold tracking-wide rounded transition-all flex items-center gap-2
                        ${isAnalyzing || !query.trim() 
                          ? 'bg-[#1C2A3A] text-[#64748B] cursor-not-allowed' 
                          : 'bg-[#38BDF8] text-[#070B12] hover:bg-[#38BDF8]/90 hover:shadow-[0_0_15px_rgba(56,189,248,0.2)]'
                        }`}
                    >
                      {isAnalyzing ? (
                        <>
                          <span className="w-3 h-3 border-2 border-[#64748B] border-t-[#070B12] rounded-full animate-spin"></span>
                          ANALYZING
                        </>
                      ) : (
                        <>ANALYZE <ChevronRight className="w-3 h-3" /></>
                      )}
                    </button>
                  </div>
                  
                  {error && (
                    <div className="mt-3 text-xs text-[#FB7185] bg-[#FB7185]/10 p-2 rounded border border-[#FB7185]/20 flex gap-2 items-start">
                      <AlertTriangle className="w-3 h-3 shrink-0 mt-0.5" />
                      {error}
                    </div>
                  )}

                  {/* Execution Workflow (Visible only while analyzing) */}
                  {isAnalyzing && (
                    <div className="mt-4 p-3 border border-[#1C2A3A] bg-[#070B12] rounded text-xs font-mono text-[#94A3B8] space-y-2">
                      <div className="flex items-center gap-2 text-[#34D399]"><CheckCircle2 className="w-3 h-3"/> Understanding request</div>
                      <div className="flex items-center gap-2 text-[#34D399]"><CheckCircle2 className="w-3 h-3"/> Validating spatial data</div>
                      <div className="flex items-center gap-2 text-[#38BDF8] animate-pulse"><span className="w-3 h-3 border-2 border-[#38BDF8] border-t-transparent rounded-full animate-spin"></span> Routing to specialist models</div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </aside>

      </main>

      {/* BOTTOM METRICS PANEL */}
      <footer className="grid grid-cols-1 md:grid-cols-4 gap-4 p-4 pt-0 shrink-0 h-[280px]">
        
        {/* 1. ANALYSIS OVERVIEW */}
        <div className="bg-[#0D1420] border border-[#1C2A3A] rounded-md p-4 flex flex-col relative overflow-hidden">
          <div className="text-[10px] font-mono tracking-widest text-[#64748B] uppercase mb-4">Analysis Result</div>
          
          {result ? (
            <div className="flex flex-col h-full">
              <div className="text-xs font-mono text-[#38BDF8] mb-2">{result.task.toUpperCase()}</div>
              <p className="text-sm text-[#F1F5F9] leading-relaxed line-clamp-4">{result.answer}</p>
              
              <div className="mt-auto flex items-center gap-2 text-[10px] font-mono text-[#34D399] pt-4 border-t border-[#1C2A3A]">
                <div className="w-1.5 h-1.5 rounded-full bg-[#34D399]"></div>
                ANALYSIS COMPLETE
              </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-xs text-[#64748B] font-mono">Awaiting execution...</div>
          )}
        </div>

        {/* 2. CONFIDENCE & STATS */}
        <div className="bg-[#0D1420] border border-[#1C2A3A] rounded-md p-4 flex flex-col">
          <div className="text-[10px] font-mono tracking-widest text-[#64748B] uppercase mb-4">System Confidence</div>
          
          {result ? (
            <>
              <div className="text-4xl font-light text-[#F1F5F9] tracking-tight mb-2">
                {result.confidence.toFixed(1)}<span className="text-lg text-[#64748B]">%</span>
              </div>
              <div className="w-full h-1 bg-[#070B12] rounded-full overflow-hidden mb-2">
                <div className="h-full bg-[#34D399]" style={{ width: `${result.confidence}%` }}></div>
              </div>
              <div className="text-[10px] font-mono text-[#94A3B8] mb-auto uppercase">Verified Output</div>
              
              {/* Secondary Stats */}
              {Object.keys(result.evidence.stats).length > 0 && (
                <div className="mt-4 pt-4 border-t border-[#1C2A3A] space-y-2">
                  {Object.entries(result.evidence.stats).map(([k, v]) => (
                    <div key={k} className="flex justify-between items-center text-xs">
                      <span className="text-[#64748B]">{k}</span>
                      <span className="font-mono text-[#F1F5F9]">{v as React.ReactNode}</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
             <div className="flex-1 flex items-center justify-center text-xs text-[#64748B] font-mono">--%</div>
          )}
        </div>

        {/* 3. SPATIAL EVIDENCE */}
        <div className="bg-[#0D1420] border border-[#1C2A3A] rounded-md p-4 flex flex-col">
          <div className="flex justify-between items-center mb-4">
            <div className="text-[10px] font-mono tracking-widest text-[#64748B] uppercase">Spatial Evidence</div>
            {result?.evidence?.regions && (
              <div className="text-[10px] font-mono bg-[#111A27] px-2 py-0.5 rounded text-[#38BDF8] border border-[#1C2A3A]">
                {result.evidence.regions.length} FEATURES
              </div>
            )}
          </div>
          
          <div className="flex-1 overflow-y-auto pr-2 space-y-2">
            {!result ? (
              <div className="h-full flex items-center justify-center text-xs text-[#64748B] font-mono">No verified regions...</div>
            ) : result.evidence.regions.length > 0 ? (
              result.evidence.regions.map((reg, idx) => (
                <div key={idx} className="bg-[#070B12] border border-[#1C2A3A] rounded p-2.5">
                  <div className="text-[10px] font-mono text-[#64748B] mb-1">REGION 0{idx + 1}</div>
                  <div className="text-xs text-[#F1F5F9] truncate">{reg.label}</div>
                  <div className="text-[10px] text-[#38BDF8] font-mono mt-1">Coverage: {reg.actual_pct.toFixed(2)}%</div>
                </div>
              ))
            ) : (
               <div className="h-full flex flex-col items-center justify-center text-center text-xs text-[#64748B] font-mono p-4">
                 <Map className="w-6 h-6 mb-2 opacity-50" />
                 No spatial boundaries matched the requested parameter.
               </div>
            )}
          </div>

          {/* Export Action */}
          {result?.gis_export && (
            <button 
              onClick={handleDownloadGIS}
              className="mt-3 w-full flex items-center justify-center gap-2 bg-[#111A27] hover:bg-[#1C2A3A] border border-[#1C2A3A] text-xs font-mono text-[#94A3B8] hover:text-[#F1F5F9] py-2 rounded transition-colors"
            >
              <Download className="w-3.5 h-3.5" /> EXPORT GEOJSON
            </button>
          )}
        </div>

        {/* 4. ANALYSIS TRACE */}
        <div className="bg-[#0D1420] border border-[#1C2A3A] rounded-md p-4 flex flex-col">
          <div className="text-[10px] font-mono tracking-widest text-[#64748B] uppercase mb-4 flex justify-between">
            <span>Execution Trace</span>
            <Activity className="w-3 h-3" />
          </div>
          
          <div className="flex-1 bg-[#070B12] border border-[#1C2A3A] rounded overflow-y-auto p-3" ref={scrollRef}>
            {!result ? (
               <div className="h-full flex items-center justify-center text-xs text-[#64748B] font-mono">System idle...</div>
            ) : (
               <div className="flex flex-col">
                 {result.trace.map((step, idx) => renderTraceLog(step, idx))}
               </div>
            )}
          </div>
        </div>

      </footer>
    </div>
  );
}