'use client';

import { useAppStore } from '@/lib/store';
import { SAMPLE_PROMPTS } from '@/types/api';
import { Wand2 } from 'lucide-react';

export function PromptInput() {
  const prompt = useAppStore((state) => state.prompt);
  const setPrompt = useAppStore((state) => state.setPrompt);

  return (
    <div className="bg-white rounded-2xl shadow-card border border-slate-100 p-6">
      <h2 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
        <Wand2 className="text-primary-500" size={20} />
        テキストから生成
      </h2>

      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="作りたいものを日本語で説明してください..."
        className="w-full h-28 bg-slate-50 border border-slate-200 rounded-xl p-4 text-slate-900
                   placeholder-slate-400 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20
                   outline-none resize-none transition-all"
      />

      <p className="text-slate-500 text-sm mt-4 mb-3">クイックスタート:</p>
      <div className="flex flex-wrap gap-2">
        {SAMPLE_PROMPTS.map((sample, index) => (
          <button
            key={index}
            onClick={() => setPrompt(sample)}
            className="bg-slate-100 hover:bg-primary-50 border border-slate-200 hover:border-primary-300
                       rounded-lg px-3 py-2 text-sm text-slate-600 hover:text-primary-700 transition-all"
          >
            {sample.slice(0, 20)}...
          </button>
        ))}
      </div>
    </div>
  );
}
