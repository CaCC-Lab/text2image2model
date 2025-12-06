'use client';

import { CheckCircle, AlertCircle, Clock } from 'lucide-react';
import { useAppStore } from '@/lib/store';

export function StatusBar() {
  const processingTime = useAppStore((state) => state.processingTime);
  const error = useAppStore((state) => state.error);

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-center gap-3">
        <div className="p-2 bg-red-100 rounded-full">
          <AlertCircle className="text-red-600" size={18} />
        </div>
        <div>
          <p className="text-red-700 font-medium">エラー</p>
          <p className="text-red-600 text-sm">{error}</p>
        </div>
      </div>
    );
  }

  if (processingTime) {
    return (
      <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-4 flex items-center gap-3">
        <div className="p-2 bg-emerald-100 rounded-full">
          <CheckCircle className="text-emerald-600" size={18} />
        </div>
        <div>
          <p className="text-emerald-700 font-medium">生成完了</p>
          <p className="text-emerald-600 text-sm">
            処理時間: {processingTime.toFixed(1)}秒
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 flex items-center gap-3">
      <div className="p-2 bg-slate-200 rounded-full">
        <Clock className="text-slate-500" size={18} />
      </div>
      <p className="text-slate-600">生成準備完了</p>
    </div>
  );
}
