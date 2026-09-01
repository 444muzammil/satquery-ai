export default function Home() {
  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-200 p-4 gap-4 font-sans">
      
      {/* HEADER (Optional but good for product feel) */}
      <header className="flex justify-between items-center pb-2 border-b border-slate-800">
        <h1 className="text-2xl font-bold text-white tracking-tight">SatQuery AI</h1>
        <span className="text-xs bg-slate-800 px-2 py-1 rounded text-slate-400">SIH26167 Prototype</span>
      </header>

      {/* TOP SECTION: 3 Columns */}
      <div className="flex flex-1 gap-4 overflow-hidden">
        
        {/* LEFT PANEL: DATA */}
        <aside className="w-72 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col gap-6">
          <h2 className="text-sm font-bold text-slate-400 tracking-wider">DATA</h2>
          
          <div className="flex flex-col gap-2">
            <label className="text-sm">Image A (Optical 1)</label>
            <button className="bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg py-6 text-sm flex items-center justify-center transition-colors">
              + Upload
            </button>
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-sm">Image B (Optical 2)</label>
            <button className="bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg py-6 text-sm flex items-center justify-center transition-colors">
              + Upload
            </button>
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-sm">SAR Image</label>
            <button className="bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg py-6 text-sm flex items-center justify-center transition-colors">
              + Upload
            </button>
          </div>
        </aside>

        {/* MIDDLE PANEL: SATELLITE VIEWER */}
        <main className="flex-1 bg-black border border-slate-800 rounded-xl flex items-center justify-center relative overflow-hidden">
          <div className="text-slate-600 flex flex-col items-center">
            <svg className="w-12 h-12 mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
            <p className="text-sm font-medium">Satellite Image Viewer</p>
          </div>
        </main>

        {/* RIGHT PANEL: AI CHAT */}
        <aside className="w-80 bg-slate-900 border border-slate-800 rounded-xl flex flex-col">
          <div className="p-4 border-b border-slate-800">
            <h2 className="text-sm font-bold text-slate-400 tracking-wider">AI ASSISTANT</h2>
          </div>
          
          <div className="flex-1 p-4 overflow-y-auto flex flex-col gap-4">
             {/* Chat history will go here later */}
             <div className="bg-slate-800 p-3 rounded-lg text-sm text-slate-300 w-11/12">
               Welcome to SatQuery AI. Upload imagery to begin analysis.
             </div>
          </div>

          <div className="p-4 border-t border-slate-800 bg-slate-900 rounded-b-xl flex flex-col gap-2">
            <textarea 
              className="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-sm focus:outline-none focus:border-blue-500 resize-none"
              rows={3}
              placeholder="Ask about your imagery..."
            ></textarea>
            <button className="w-full bg-blue-600 hover:bg-blue-500 text-white font-medium py-2 rounded-lg transition-colors">
              Analyze
            </button>
          </div>
        </aside>

      </div>

      {/* BOTTOM SECTION: METRICS */}
      <div className="h-40 flex gap-4">
        <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
          <h3 className="text-xs font-bold text-slate-400 tracking-wider mb-2">ANALYSIS</h3>
          <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">Awaiting query...</div>
        </div>
        <div className="w-48 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
          <h3 className="text-xs font-bold text-slate-400 tracking-wider mb-2">CONFIDENCE</h3>
          <div className="flex-1 flex items-center justify-center text-3xl font-light text-slate-600">--%</div>
        </div>
        <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
          <h3 className="text-xs font-bold text-slate-400 tracking-wider mb-2">EVIDENCE</h3>
          <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">No regions detected</div>
        </div>
        <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
          <h3 className="text-xs font-bold text-slate-400 tracking-wider mb-2">EXECUTION TRACE</h3>
          <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">System idle</div>
        </div>
      </div>

    </div>
  );
}