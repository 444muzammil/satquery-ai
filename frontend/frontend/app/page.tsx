"use client";

import React, { useState, useRef, useEffect } from 'react';
import { 
  Satellite, Crosshair, Upload, Map, Layers, Target, Activity, 
  CheckCircle2, AlertTriangle, ShieldCheck, Download, 
  ChevronRight, Database, Maximize, Minimize, Cpu, Sparkles, Trash2, LayoutDashboard, MessageSquare, Eye, EyeOff, Scan, Loader2, CornerDownRight
} from 'lucide-react';

// =====================================================================
// TYPES & INTERFACES
// =====================================================================
interface Region {
  box: [number, number, number, number]; // [ymin_pct, xmin_pct, ymax_pct, xmax_pct]
  polygon?: [number, number][]; // [x_pct, y_pct][]
  label: string;
  actual_pct: number;
}

interface AnalysisResult {
  answer: string;
  short_summary?: string;
  confidence: number;
  task: string;
  model_used: string;
  tools_executed?: string[];
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
  const [imageA, setImageA] = useState<ImageState>(defaultImageState);
  const [imageB, setImageB] = useState<ImageState>(defaultImageState);
  const [imageSAR, setImageSAR] = useState<ImageState>(defaultImageState);

  const [activeTab, setActiveTab] = useState<'A' | 'B' | 'SAR'>('A');
  const [rightTab, setRightTab] = useState<'CHAT' | 'DETAILS'>('CHAT');
  const [showMarkings, setShowMarkings] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const [query, setQuery] = useState("");
  const [lastExecutedQuery, setLastExecutedQuery] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingPhrase, setLoadingPhrase] = useState<string>('');
  
  const fileInputA = useRef<HTMLInputElement>(null);
  const fileInputB = useRef<HTMLInputElement>(null);
  const fileInputSAR = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const copilotScrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
    if (copilotScrollRef.current) {
      copilotScrollRef.current.scrollTop = copilotScrollRef.current.scrollHeight;
    }
  }, [result, rightTab, isAnalyzing]);

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
      setActiveTab(type);
      setResult(null);
    } catch (err) {
      console.error(err);
      setFn({ file: null, preview: "", metadata: {}, status: 'error' });
    }
  };

  const handleRemove = (type: 'A' | 'B' | 'SAR') => {
    const setFn = type === 'A' ? setImageA : type === 'B' ? setImageB : setImageSAR;
    setFn(defaultImageState);
    if (type === 'A') setResult(null);
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
    setRightTab('CHAT'); 
    setLastExecutedQuery(query);
    
    let phrases = [
      "Calibrating Multispectral Sensors...",
      "Extracting VLM Semantic Priors...",
      "Running Agentic GrabCut Algorithm...",
      "Applying Morphological Closings...",
      "Isolating Spatial Boundaries...",
      "Synthesizing Intelligence Brief..."
    ];
    
    const q = query.toLowerCase();
    if (q.includes('change') || q.includes('new') || q.includes('construct')) {
      phrases = [
        "Aligning Bi-Temporal Telemetry...",
        "Executing Spatial Pixel Differencing...",
        "Applying Structural Morphological Closings...",
        "Generating Change Polygons...",
        "Synthesizing Geospatial Brief..."
      ];
    } else if (q.includes('sar') || q.includes('fusion') || q.includes('radar')) {
      phrases = [
        "Co-registering Optical and SAR Datasets...",
        "Fusing Microwave Backscatter Signals...",
        "Isolating Structural Density...",
        "Mapping Multimodal Features...",
        "Synthesizing Intelligence Brief..."
      ];
    }

    let step = 0;
    setLoadingPhrase(phrases[0]);
    const phraseInterval = setInterval(() => {
      step++;
      if (step < phrases.length) setLoadingPhrase(phrases[step]);
      // Stop looping at the end so it stays on the "Synthesizing..." text instead of jumping back
    }, 1500);

    const startTime = Date.now();

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
      
      const elapsed = Date.now() - startTime;
      if (elapsed < 4800) {
        await new Promise(r => setTimeout(r, 4800 - elapsed));
      }
      
      setResult(data);
      setShowMarkings(true);
    } catch (err: any) {
      setError(err.message || "An error occurred during analysis.");
    } finally {
      clearInterval(phraseInterval);
      setIsAnalyzing(false);
    }
  };

  const getSuggestions = () => {
    if (imageSAR.preview && imageA.preview) return [
      { text: "Analyze structural density using optical-SAR fusion", q: "Analyze structural density using optical-SAR fusion." },
      { text: "Detect water bodies using radar backscatter", q: "Detect water bodies using radar backscatter." }
    ];
    if (imageB.preview && imageA.preview) return [
      { text: "Detect new construction and infrastructure", q: "Detect new construction and infrastructure." },
      { text: "Compare structural development between dates", q: "Compare structural development between dates." },
      { text: "Highlight barren land changes", q: "Highlight barren land changes." }
    ];
    return [
      { text: "Locate the major water body", q: "Locate the major water body." },
      { text: "Mark the vegetation and forests", q: "Mark the vegetation and forests." },
      { text: "Analyze the structural development density", q: "Analyze the structural development density." },
      { text: "Highlight barren land and soil", q: "Highlight barren land and soil." }
    ];
  };

  const getFollowUpSuggestions = (lastQ: string) => {
    if (!lastQ) return [];
    const q = lastQ.toLowerCase();
    if (q.includes('water') || q.includes('river') || q.includes('coastline') || q.includes('marine')) return [
      { text: "Calculate the exact spatial area of the water body", q: "Calculate the exact spatial area of the water body." },
      { text: "Highlight nearby structural development", q: "Highlight nearby structural development." }
    ];
    if (q.includes('change') || q.includes('new') || q.includes('construct') || q.includes('develop')) return [
      { text: "Calculate the total area of the new construction", q: "Calculate the total area of the new construction." },
      { text: "Highlight the surrounding vegetation", q: "Highlight the surrounding vegetation." }
    ];
    if (q.includes('vegetation') || q.includes('forest')) return [
      { text: "Calculate the total vegetated area", q: "Calculate the total vegetated area." },
      { text: "Highlight barren land nearby", q: "Highlight barren land nearby." }
    ];
    return [
      { text: "Calculate the spatial area of these features", q: "Calculate the spatial area of these features." },
      { text: "Describe the overall scene in detail", q: "Describe the overall scene in detail." }
    ];
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

  const handleDownloadReport = () => {
    if (!result?.report_data) return;
    const blob = new Blob([JSON.stringify(result.report_data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `SatQuery_Report_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const renderTraceLog = (traceString: string, idx: number) => {
    const match = traceString.match(/^\[(.*?)\]\s*(.*)$/);
    if (!match) return <div key={idx} className="text-slate-500 font-mono text-xs py-1">{traceString}</div>;
    
    const tag = match[1];
    const content = match[2];
    
    let tagColor = "text-blue-800"; // ISRO Blue
    if (tag === "OBSERVATION" || tag === "VERIFICATION") tagColor = "text-emerald-600";
    if (tag === "COMPATIBILITY" || tag === "INPUT_VALIDATION") tagColor = "text-slate-500";
    if (tag === "FALLBACK") tagColor = "text-orange-500"; // Saffron
    if (content.includes("FAILED") || tag.includes("REJECTED")) tagColor = "text-rose-600";

    return (
      <div key={idx} className="font-mono text-xs mb-3 border-l-2 border-slate-200 pl-3 py-0.5 animate-in fade-in duration-300">
        <div className={`font-semibold tracking-wide ${tagColor}`}>{tag.replace(/_/g, ' ')}</div>
        <div className="text-slate-600 mt-1 leading-relaxed">{content}</div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans flex flex-col selection:bg-orange-100 selection:text-orange-900 transition-colors duration-500">
      <style dangerouslySetInnerHTML={{__html: `
        @keyframes drawPolygon {
          0% { opacity: 0; fill-opacity: 0; }
          50% { opacity: 1; fill-opacity: 0; }
          100% { opacity: 1; fill-opacity: 0.2; }
        }
        .animate-draw-polygon {
          opacity: 0;
          fill-opacity: 0;
          animation: drawPolygon 1.5s ease-out forwards;
        }
        @keyframes tagFadeIn {
          0% { opacity: 0; transform: translateY(-80%) scale(0.9); }
          100% { opacity: 1; transform: translateY(-120%) scale(1); }
        }
        .animate-tag-in {
          opacity: 0;
          animation: tagFadeIn 0.5s ease-out forwards;
        }
      `}} />
      {/* HEADER: ISRO Heritage Theme */}
      <header className="flex items-center justify-between px-6 py-4 bg-white shadow-[0_4px_20px_rgb(0,0,0,0.03)] z-10 shrink-0 border-b border-slate-100">
        <div className="flex items-center gap-3">
          <div className="bg-blue-50 p-2 rounded-xl">
            <Satellite className="w-6 h-6 text-blue-800" />
          </div>
          <div>
            <h1 className="text-lg font-black tracking-widest text-blue-900">SATQUERY <span className="text-orange-500">AI</span></h1>
            <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">Remote Sensing Intelligence</p>
          </div>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-2 text-emerald-600 font-bold bg-emerald-50 px-3 py-1.5 rounded-full border border-emerald-100 shadow-sm">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            SYSTEM READY
          </div>
          <span className="px-3 py-1.5 rounded-full bg-blue-50 text-blue-800 font-bold border border-blue-100 shadow-sm">
            SIH26167
          </span>
        </div>
      </header>

      {/* MAIN CONTENT AREA */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 min-h-0 bg-slate-50/50">
        
        {/* LEFT PANEL: DATA INPUTS */}
        <aside className="lg:col-span-3 flex flex-col gap-4 h-full">
          <div className="bg-white border border-slate-100 rounded-2xl flex flex-col h-full shadow-[0_8px_30px_rgb(0,0,0,0.04)] overflow-hidden transition-all duration-300">
            <div className="p-5 border-b border-slate-100 flex items-center gap-2 bg-white z-10">
              <Database className="w-5 h-5 text-slate-400" />
              <h2 className="text-sm font-bold tracking-widest text-slate-800 uppercase">Data Inputs</h2>
            </div>
            
            <div className="p-5 flex flex-col gap-5 overflow-y-auto">
              {[
                { title: "OPTICAL / BASE", label: "Image A", state: imageA, ref: fileInputA, type: 'A', activeColor: "bg-blue-800" },
                { title: "BI-TEMPORAL", label: "Image B", state: imageB, ref: fileInputB, type: 'B', activeColor: "bg-indigo-600" },
                { title: "RADAR / FUSION", label: "SAR Data", state: imageSAR, ref: fileInputSAR, type: 'SAR', activeColor: "bg-orange-500" }
              ].map((inp) => (
                <div key={inp.type} className="border border-slate-100 rounded-xl bg-slate-50/50 p-4 shadow-sm hover:shadow-md hover:border-blue-200 hover:bg-white transition-all duration-300 group">
                  <div className="flex justify-between items-center mb-3">
                    <span className="text-[10px] font-bold tracking-widest text-slate-500 flex items-center gap-2 uppercase">
                      <span className={`w-2 h-2 ${inp.activeColor} rounded-full shadow-sm`}></span> {inp.title}
                    </span>
                    {inp.state.status === 'loaded' && <CheckCircle2 className="w-5 h-5 text-emerald-500" />}
                  </div>
                  
                  <div className="mb-4 text-sm text-slate-800 font-bold">{inp.label}</div>
                  
                  {inp.state.status === 'loaded' ? (
                    <div className="space-y-3 animate-in fade-in duration-300">
                      <div className="text-xs text-slate-600 font-mono truncate bg-white p-2.5 rounded-lg border border-slate-200 shadow-inner">
                        {inp.state.metadata.filename || "satellite_scene.tif"}
                      </div>
                      <div className="flex gap-2 text-[10px] text-slate-500 font-mono flex-wrap">
                        {inp.state.metadata.modality && (
                          <>
                            <span className="text-emerald-600 font-bold">{inp.state.metadata.modality.toUpperCase()}</span>
                            <span>•</span>
                          </>
                        )}
                        <span>{inp.state.metadata.width}×{inp.state.metadata.height}</span>
                        <span>•</span>
                        <span>{inp.state.metadata.bands} BANDS</span>
                        <span>•</span>
                        <span>{inp.state.metadata.crs || "EPSG:4326"}</span>
                      </div>
                      <div className="mt-4 flex gap-2">
                        <button 
                          onClick={() => inp.ref.current?.click()}
                          className="flex-1 text-xs py-2 font-bold border border-slate-200 bg-white text-slate-600 hover:text-blue-800 hover:border-blue-200 transition-all rounded-lg shadow-sm hover:shadow active:scale-95"
                        >
                          Replace
                        </button>
                        <button 
                          onClick={() => handleRemove(inp.type as 'A' | 'B' | 'SAR')}
                          title="Remove dataset"
                          className="px-3 text-xs py-2 border border-slate-200 bg-white text-rose-500 hover:bg-rose-50 hover:border-rose-200 transition-all rounded-lg shadow-sm hover:shadow flex items-center justify-center active:scale-95"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  ) : inp.state.status === 'loading' ? (
                    <div className="text-xs font-bold font-mono text-blue-800 animate-pulse flex items-center gap-2 py-4">
                      <Activity className="w-4 h-4" /> Ingesting telemetry...
                    </div>
                  ) : (
                    <button 
                      onClick={() => inp.ref.current?.click()}
                      className="w-full border-2 border-dashed border-slate-200 hover:border-orange-400 bg-white hover:bg-orange-50 text-slate-400 hover:text-orange-600 text-xs py-6 rounded-xl transition-all duration-300 flex flex-col items-center gap-3 group-hover:-translate-y-0.5"
                    >
                      <Upload className="w-5 h-5 transition-transform group-hover:scale-110" />
                      <span className="font-bold tracking-wide">Upload {inp.label}</span>
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

        {/* CENTER PANEL: SATELLITE VIEWER */}
        <section className="lg:col-span-6 flex flex-col bg-white border border-slate-100 rounded-2xl overflow-hidden relative shadow-[0_8px_30px_rgb(0,0,0,0.04)] h-full">
          <div className="flex bg-slate-50/80 backdrop-blur border-b border-slate-100">
            {[
              { id: 'A', label: 'IMAGE A', active: !!imageA.preview },
              { id: 'B', label: 'IMAGE B', active: !!imageB.preview },
              { id: 'SAR', label: 'SAR VIEW', active: !!imageSAR.preview }
            ].map((tab) => (
              <button
                key={tab.id}
                disabled={!tab.active}
                onClick={() => setActiveTab(tab.id as 'A' | 'B' | 'SAR')}
                className={`flex-1 py-4 text-xs font-bold font-mono tracking-widest transition-all duration-300
                  ${activeTab === tab.id 
                    ? 'bg-white text-blue-800 border-b-2 border-b-blue-800 shadow-[0_-4px_10px_rgb(0,0,0,0.02)]' 
                    : tab.active 
                      ? 'text-slate-500 hover:bg-slate-100 hover:text-slate-800 border-b-2 border-b-transparent' 
                      : 'text-slate-300 cursor-not-allowed border-b-2 border-b-transparent'
                  }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div className="flex-1 relative bg-slate-100/50 overflow-hidden flex items-center justify-center">
            {(!imageA.preview && !imageB.preview && !imageSAR.preview) && (
              <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none p-6 text-center animate-in fade-in duration-700">
                <div className="absolute inset-0 opacity-[0.02]" style={{ backgroundImage: 'radial-gradient(#000080 1px, transparent 1px)', backgroundSize: '32px 32px' }}></div>
                <Crosshair className="w-16 h-16 text-slate-300 mb-6 drop-shadow-sm" />
                <h3 className="text-slate-800 font-black tracking-widest mb-2 text-xl">SATQUERY <span className="text-orange-500">AI</span></h3>
                <p className="text-slate-500 text-sm mb-6 max-w-xs font-medium">Interactive Geospatial Workspace</p>
                <div className="px-5 py-2.5 border border-slate-200 rounded-full bg-white shadow-sm text-slate-500 text-xs font-mono font-bold">
                  Ingest telemetry from Data Inputs
                </div>
              </div>
            )}

            {((activeTab === 'A' && imageA.preview) || 
              (activeTab === 'B' && imageB.preview) || 
              (activeTab === 'SAR' && imageSAR.preview)) && (
              <div className={`relative w-full h-full flex items-center justify-center group overflow-auto transition-all duration-300 ${isFullscreen ? 'fixed inset-0 z-50 bg-slate-900/95 p-12 backdrop-blur-sm' : 'p-6'}`}>
                <div className={`relative inline-block border-4 border-white bg-white rounded-md transition-all ${isFullscreen ? 'max-w-[95vw] max-h-[95vh] shadow-[0_0_100px_rgba(0,0,0,0.5)]' : 'max-w-full max-h-full shadow-xl shadow-slate-300/50'}`}>
                  <img 
                    src={activeTab === 'A' ? imageA.preview : activeTab === 'B' ? imageB.preview : imageSAR.preview} 
                    alt="Satellite Observation" 
                    className={`max-w-full max-h-full object-contain transition-all duration-700 ${isAnalyzing ? 'blur-md brightness-50 scale-105' : ''}`}
                  />
                  
                  {isAnalyzing && (
                    <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-slate-900/40 backdrop-blur-sm">
                      <div className="relative w-24 h-24 mb-6">
                        <div className="absolute inset-0 border-4 border-emerald-500/20 rounded-full"></div>
                        <div className="absolute inset-0 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
                        <div className="absolute inset-3 border-4 border-orange-500/20 rounded-full"></div>
                        <div className="absolute inset-3 border-4 border-orange-500 border-b-transparent rounded-full animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1.2s' }}></div>
                        <Scan className="absolute inset-0 m-auto w-8 h-8 text-emerald-400 animate-pulse" />
                      </div>
                      <div className="px-4 py-2 bg-slate-900/80 border border-slate-700 rounded-lg shadow-2xl">
                        <p className="text-emerald-400 font-mono text-xs tracking-widest uppercase animate-pulse">
                          {loadingPhrase || "INITIALIZING VLM..."}
                        </p>
                      </div>
                    </div>
                  )}
                  
                  {showMarkings && result?.evidence?.regions?.map((reg, idx) => {
                    const [ymin, xmin, ymax, xmax] = reg.box;
                    const isWater = reg.label.toLowerCase().includes("water") || reg.label.toLowerCase().includes("flood");
                    const isChange = reg.label.toLowerCase().includes("change") || reg.label.toLowerCase().includes("new");
                    
                    // Thematic border colors
                    let borderColor = "border-emerald-500";
                    let bgColor = "bg-emerald-500/20";
                    let tagColor = "bg-emerald-500 text-white";
                    let strokeColor = "#10b981"; // emerald-500

                    if (isWater) {
                      borderColor = "border-blue-500";
                      bgColor = "bg-blue-500/20";
                      tagColor = "bg-blue-500 text-white";
                      strokeColor = "#3b82f6"; // blue-500
                    } else if (isChange || activeTab === 'SAR') {
                      borderColor = "border-orange-500";
                      bgColor = "bg-orange-500/20";
                      tagColor = "bg-orange-500 text-white";
                      strokeColor = "#f97316"; // orange-500
                    }
                    
                    return (
                      <React.Fragment key={idx}>
                        {reg.polygon && reg.polygon.length > 2 ? (
                          <svg viewBox="0 0 100 100" preserveAspectRatio="none" overflow="visible" className="absolute inset-0 w-full h-full pointer-events-none z-10">
                            <polygon 
                              points={reg.polygon.map(pt => `${pt[0]},${pt[1]}`).join(' ')}
                              fill={strokeColor}
                              stroke={strokeColor}
                              strokeWidth="2"
                              vectorEffect="non-scaling-stroke"
                              className="animate-draw-polygon"
                              style={{ animationDelay: `${idx * 0.15}s` }}
                            />
                          </svg>
                        ) : (
                          <div 
                            className={`absolute border-2 ${borderColor} ${bgColor} pointer-events-none transition-all duration-500 ease-out z-10`}
                            style={{
                              top: `${ymin}%`,
                              left: `${xmin}%`,
                              height: `${ymax - ymin}%`,
                              width: `${xmax - xmin}%`
                            }}
                          />
                        )}
                        
                        <div 
                          className={`absolute px-2 py-0.5 text-[9px] font-bold font-mono whitespace-nowrap shadow-md rounded-md pointer-events-none z-20 animate-tag-in ${tagColor}`}
                          style={{
                            top: `${ymin}%`,
                            left: `${xmin}%`,
                            animationDelay: `${idx * 0.15 + 0.5}s`
                          }}
                        >
                          {reg.label.toUpperCase()}
                        </div>
                      </React.Fragment>
                    );
                  })}
                </div>
                
                <div className="absolute top-6 right-6 flex flex-col gap-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-50">
                  {result?.evidence?.regions && result.evidence.regions.length > 0 && (
                    <button 
                      onClick={() => setShowMarkings(!showMarkings)}
                      title={showMarkings ? "Hide Regions" : "Show Regions"}
                      className="p-3 bg-white/90 backdrop-blur-md shadow-lg border border-slate-200 rounded-xl text-slate-600 hover:text-orange-600 hover:border-orange-300 hover:scale-105 transition-all"
                    >
                      {showMarkings ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  )}
                  <button 
                    onClick={() => setIsFullscreen(!isFullscreen)}
                    title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
                    className="p-3 bg-white/90 backdrop-blur-md shadow-lg border border-slate-200 rounded-xl text-slate-600 hover:text-blue-800 hover:border-blue-300 hover:scale-105 transition-all"
                  >
                    {isFullscreen ? <Minimize className="w-4 h-4" /> : <Maximize className="w-4 h-4" />}
                  </button>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* RIGHT PANEL: COPILOT & DETAILS (TABS) */}
        <aside className="lg:col-span-3 flex flex-col gap-4 h-full">
          <div className="bg-white border border-slate-100 rounded-2xl flex flex-col h-full shadow-[0_8px_30px_rgb(0,0,0,0.04)] overflow-hidden transition-all duration-300">
            {/* Header Tabs */}
            <div className="flex bg-slate-50/80 backdrop-blur border-b border-slate-100 shrink-0">
              <button
                onClick={() => setRightTab('CHAT')}
                className={`flex-1 flex items-center justify-center gap-2 py-4 text-xs font-bold tracking-widest transition-all duration-300
                  ${rightTab === 'CHAT' 
                    ? 'bg-white text-blue-800 border-b-2 border-b-blue-800 shadow-[0_-4px_10px_rgb(0,0,0,0.02)]' 
                    : 'text-slate-500 hover:bg-slate-100 hover:text-slate-800 border-b-2 border-b-transparent'
                  }`}
              >
                <MessageSquare className="w-4 h-4" />
                COPILOT
              </button>
              <button
                onClick={() => setRightTab('DETAILS')}
                className={`flex-1 flex items-center justify-center gap-2 py-4 text-xs font-bold tracking-widest transition-all duration-300
                  ${rightTab === 'DETAILS' 
                    ? 'bg-white text-orange-600 border-b-2 border-b-orange-600 shadow-[0_-4px_10px_rgb(0,0,0,0.02)]' 
                    : 'text-slate-500 hover:bg-slate-100 hover:text-slate-800 border-b-2 border-b-transparent'
                  }`}
              >
                <LayoutDashboard className="w-4 h-4" />
                DETAILS
              </button>
            </div>

            {/* TAB CONTENT: CHAT */}
            {rightTab === 'CHAT' && (
              <div className="p-5 flex-1 flex flex-col overflow-y-auto" ref={copilotScrollRef}>
                {!imageA.preview ? (
                  <div className="text-center my-auto animate-in fade-in duration-700">
                    <div className="w-14 h-14 rounded-2xl bg-blue-50 border border-blue-100 flex items-center justify-center mx-auto mb-4 text-blue-800 shadow-inner">
                      <Target className="w-6 h-6" />
                    </div>
                    <p className="text-slate-800 font-black mb-1">Awaiting Telemetry</p>
                    <p className="text-xs text-slate-500 font-medium">Load base imagery to initiate Copilot.</p>
                  </div>
                ) : (
                  <div className="flex flex-col h-full justify-between gap-5">
                    
                    {/* Suggestions Box */}
                    {!result && !isAnalyzing && (
                      <div className="space-y-3 animate-in fade-in slide-in-from-bottom-4 duration-500">
                        <p className="text-[10px] font-bold font-mono text-slate-400 uppercase tracking-widest mb-2 flex items-center gap-1.5">
                          <Sparkles className="w-3.5 h-3.5 text-orange-400" /> Suggested Tasks
                        </p>
                        {getSuggestions().map((sug, idx) => (
                          <button 
                            key={idx} 
                            onClick={() => setQuery(sug.q)} 
                            className="w-full text-left px-4 py-3 text-xs font-bold bg-white border border-slate-200 shadow-sm rounded-xl hover:border-orange-300 hover:bg-orange-50 hover:text-orange-700 text-slate-700 transition-all hover:-translate-y-0.5 truncate"
                          >
                            {sug.text}
                          </button>
                        ))}
                      </div>
                    )}

                    {/* ACTIVE INTELLIGENCE OUTPUT BUBBLE */}
                    {result && (
                      <>
                        <div className="bg-gradient-to-br from-blue-50 to-white border border-blue-100 rounded-2xl p-5 relative shadow-md shadow-blue-900/5 animate-in fade-in slide-in-from-bottom-4 duration-500">
                          <div className="flex items-center justify-between mb-4">
                            <span className="text-[10px] font-black font-mono tracking-widest text-blue-800 uppercase flex items-center gap-1.5">
                              <Sparkles className="w-3.5 h-3.5 text-orange-500" />
                              Intelligence Brief
                            </span>
                            <span className="text-[10px] font-black font-mono text-emerald-700 bg-emerald-100 px-2.5 py-1 rounded-full shadow-sm">
                              {result.confidence.toFixed(0)}% CONFIDENCE
                            </span>
                          </div>
                          
                          <p className="text-sm text-slate-800 leading-relaxed font-bold mb-4">
                            {result.answer}
                          </p>

                          {/* NEW: Result Summary Card */}
                          <div className="bg-white border border-slate-100 rounded-xl p-3.5 shadow-sm flex items-center justify-between">
                            <div className="flex flex-col">
                              <span className="text-[9px] font-bold font-mono text-slate-400 uppercase tracking-widest mb-1">Primary Task</span>
                              <span className="text-xs font-black text-blue-900">{result.task.replace(/_/g, ' ')}</span>
                            </div>
                            {result.short_summary && (
                              <div className="flex flex-col text-right">
                                <span className="text-[9px] font-bold font-mono text-slate-400 uppercase tracking-widest mb-1">Summary</span>
                                <span className="text-xs font-bold text-slate-600 truncate max-w-[200px]">{result.short_summary}</span>
                              </div>
                            )}
                          </div>
                          
                          <div className="mt-4 pt-3 border-t border-blue-100/50 flex justify-end">
                            <button 
                              onClick={() => setRightTab('DETAILS')}
                              className="text-xs font-bold text-orange-600 hover:text-orange-700 flex items-center gap-1 transition-colors hover:translate-x-1"
                            >
                              View Evidence & Details <ChevronRight className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        </div>

                        {/* Follow-up Suggestions */}
                        <div className="space-y-3 mt-4 animate-in fade-in slide-in-from-bottom-2 duration-500">
                          <p className="text-[10px] font-bold font-mono text-slate-400 uppercase tracking-widest mb-2 flex items-center gap-1.5">
                            <CornerDownRight className="w-3.5 h-3.5 text-orange-400" /> Suggested Follow-ups
                          </p>
                          {getFollowUpSuggestions(lastExecutedQuery).map((sug, idx) => (
                            <button 
                              key={idx} 
                              onClick={() => setQuery(sug.q)} 
                              className="w-full text-left px-4 py-3 text-xs font-bold bg-white border border-slate-200 shadow-sm rounded-xl hover:border-orange-300 hover:bg-orange-50 hover:text-orange-700 text-slate-700 transition-all hover:-translate-y-0.5 truncate"
                            >
                              {sug.text}
                            </button>
                          ))}
                        </div>
                      </>
                    )}

                    {/* Query Input Section */}
                    <div className="relative mt-auto shrink-0 pt-4">
                      <div className="relative shadow-sm rounded-2xl">
                        <textarea 
                          value={query}
                          onChange={(e) => setQuery(e.target.value)}
                          placeholder="Command your AI assistant..."
                          className="w-full bg-slate-50 border border-slate-200 rounded-2xl p-4 pb-14 text-sm text-slate-800 font-medium placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-800/20 focus:border-blue-800 focus:bg-white transition-all resize-none h-32"
                          disabled={isAnalyzing}
                        />
                        <button 
                          onClick={handleAnalyze}
                          disabled={isAnalyzing || !query.trim()}
                          className={`absolute bottom-3 right-3 px-5 py-2.5 text-xs font-black tracking-wider rounded-xl transition-all flex items-center gap-2 shadow-sm
                            ${isAnalyzing || !query.trim() 
                              ? 'bg-slate-200 text-slate-400 cursor-not-allowed' 
                              : 'bg-blue-800 text-white hover:bg-blue-900 hover:shadow-md hover:shadow-blue-900/20 active:scale-95'
                            }`}
                        >
                          {isAnalyzing ? (
                            <>
                              <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                              ANALYZING
                            </>
                          ) : (
                            <>EXECUTE <ChevronRight className="w-4 h-4" /></>
                          )}
                        </button>
                      </div>
                    </div>
                    
                    {error && (
                      <div className="text-xs text-rose-700 font-bold bg-rose-50 p-4 rounded-xl border border-rose-200 flex gap-2 items-start mt-2 shadow-sm animate-in fade-in slide-in-from-bottom-2">
                        <AlertTriangle className="w-4 h-4 shrink-0" />
                        {error}
                      </div>
                    )}

                    {isAnalyzing && (
                      <div className="p-5 border border-slate-100 bg-white rounded-2xl text-xs font-bold font-mono text-slate-600 space-y-3 shadow-md mt-2 animate-in fade-in slide-in-from-bottom-2">
                        <div className="flex items-center gap-3 text-emerald-600"><CheckCircle2 className="w-4 h-4"/> Parsing intent</div>
                        <div className="flex items-center gap-3 text-emerald-600"><CheckCircle2 className="w-4 h-4"/> Validating spatial constraints</div>
                        <div className="flex items-center gap-3 text-orange-500 animate-pulse"><span className="w-3.5 h-3.5 border-2 border-orange-500 border-t-transparent rounded-full animate-spin"></span> Routing to specialists</div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* TAB CONTENT: DETAILS (Evidence, Trace, etc) */}
            {rightTab === 'DETAILS' && (
              <div className="flex-1 flex flex-col overflow-y-auto bg-slate-50/50 p-5 gap-5 animate-in fade-in duration-500">
                {!result ? (
                   <div className="text-center my-auto">
                     <Activity className="w-8 h-8 text-slate-300 mx-auto mb-3" />
                     <p className="text-sm text-slate-500 font-bold">Run an analysis to view deep metrics.</p>
                   </div>
                ) : (
                  <>
                    {/* STATS & TOOLS */}
                    <div className="bg-white border border-slate-100 rounded-2xl p-5 shadow-sm hover:shadow-md transition-shadow">
                      <div className="text-[10px] font-black font-mono tracking-widest text-blue-900 uppercase mb-4 flex items-center gap-2">
                        Execution Profile
                      </div>
                      <div className="space-y-4">
                        {result.tools_executed && result.tools_executed.length > 0 && (
                          <div>
                            <div className="text-[10px] text-slate-400 uppercase mb-2 font-bold tracking-wider">Models Engaged:</div>
                            <div className="flex flex-wrap gap-2">
                              {result.tools_executed.map(t => (
                                <span key={t} className="px-2.5 py-1 bg-slate-100 text-slate-700 text-[10px] font-bold font-mono border border-slate-200 rounded-lg shadow-sm">
                                  {t}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        
                        {result.evidence?.stats && Object.keys(result.evidence.stats).length > 0 && (
                          <div className="pt-3 border-t border-slate-100">
                            <div className="text-[10px] text-slate-400 uppercase mb-3 font-bold tracking-wider">Calculated Metrics:</div>
                            <div className="space-y-2">
                              {Object.entries(result.evidence.stats).map(([k, v]) => (
                                <div key={k} className="flex justify-between items-center text-xs">
                                  <span className="text-slate-600 font-bold">{k}</span>
                                  <span className="font-mono font-black text-blue-800 bg-blue-50 px-2 py-0.5 rounded border border-blue-100">{String(v)}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* SPATIAL EVIDENCE */}
                    <div className="bg-white border border-slate-100 rounded-2xl p-5 shadow-sm hover:shadow-md transition-shadow flex flex-col max-h-[320px]">
                      <div className="flex justify-between items-center mb-4 shrink-0">
                        <div className="text-[10px] font-black font-mono tracking-widest text-blue-900 uppercase">Spatial Grounding</div>
                        {result.evidence?.regions && (
                          <div className="text-[10px] font-black font-mono bg-orange-100 px-2.5 py-1 rounded-full text-orange-700 border border-orange-200 shadow-sm">
                            {result.evidence.regions.length} FEATURES
                          </div>
                        )}
                      </div>
                      
                      <div className="flex-1 overflow-y-auto pr-2 space-y-2.5">
                        {(result.evidence?.regions?.length || 0) > 0 ? (
                          result.evidence.regions.map((reg, idx) => (
                            <div key={idx} className="bg-slate-50 border border-slate-200 rounded-xl p-3 hover:border-blue-300 hover:bg-white transition-all shadow-sm cursor-default">
                              <div className="flex justify-between items-center mb-1.5">
                                <div className="text-[10px] font-black font-mono text-slate-400">REGION {String(idx + 1).padStart(2, '0')}</div>
                                <div className="text-[10px] text-emerald-600 font-black font-mono">Cov: {reg.actual_pct.toFixed(2)}%</div>
                              </div>
                              <div className="text-xs font-bold text-slate-800 capitalize">{reg.label}</div>
                            </div>
                          ))
                        ) : (
                           <div className="h-full flex flex-col items-center justify-center text-center text-xs text-slate-500 font-bold p-4">
                             <Map className="w-6 h-6 mb-3 opacity-30 text-slate-400" />
                             No explicit boundaries localized.
                           </div>
                        )}
                      </div>

                      <div className="mt-4 pt-4 border-t border-slate-100 flex gap-3 shrink-0">
                        {result.gis_export && (
                          <button onClick={handleDownloadGIS} className="flex-1 flex items-center justify-center gap-2 bg-slate-100 hover:bg-blue-800 hover:text-white border border-slate-200 hover:border-blue-800 text-[10px] font-black text-slate-700 py-2.5 rounded-xl shadow-sm transition-all active:scale-95">
                            <Download className="w-3.5 h-3.5" /> GEOJSON
                          </button>
                        )}
                        {result.report_data && (
                          <button onClick={handleDownloadReport} className="flex-1 flex items-center justify-center gap-2 bg-slate-100 hover:bg-blue-800 hover:text-white border border-slate-200 hover:border-blue-800 text-[10px] font-black text-slate-700 py-2.5 rounded-xl shadow-sm transition-all active:scale-95">
                            <Download className="w-3.5 h-3.5" /> REPORT
                          </button>
                        )}
                      </div>
                    </div>

                    {/* TRACE */}
                    <div className="bg-white border border-slate-100 rounded-2xl p-5 shadow-sm hover:shadow-md transition-shadow flex flex-col flex-1 min-h-[280px]">
                      <div className="text-[10px] font-black font-mono tracking-widest text-blue-900 uppercase mb-4 flex items-center gap-2 shrink-0">
                        Execution Trace <Activity className="w-4 h-4 text-orange-500" />
                      </div>
                      
                      <div className="flex-1 bg-slate-50 border border-slate-200 rounded-xl p-4 overflow-y-auto shadow-inner" ref={scrollRef}>
                         <div className="flex flex-col">
                           {result.trace?.map((step, idx) => renderTraceLog(step, idx))}
                         </div>
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}