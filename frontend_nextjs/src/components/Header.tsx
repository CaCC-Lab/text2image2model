'use client';

import { useAppStore } from '@/lib/store';
import { Sparkles, Wifi, WifiOff } from 'lucide-react';

export function Header() {
  const backendConnected = useAppStore((state) => state.backendConnected);

  return (
    <header className="bg-white/80 backdrop-blur-md border-b border-slate-200 sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl shadow-lg shadow-primary-500/25">
            <Sparkles className="text-white" size={24} />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-900">
              AI 3D Generator
            </h1>
            <p className="text-xs text-slate-500">
              Text & Image to 3D Model
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div
            className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
              backendConnected
                ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                : 'bg-red-50 text-red-700 border border-red-200'
            }`}
          >
            {backendConnected ? (
              <>
                <Wifi size={14} />
                接続済み
              </>
            ) : (
              <>
                <WifiOff size={14} />
                未接続
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
