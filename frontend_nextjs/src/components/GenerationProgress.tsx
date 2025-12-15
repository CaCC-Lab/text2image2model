'use client';

import { useEffect, useState } from 'react';
import { Loader2, Image as ImageIcon, Box, Palette, Download, CheckCircle2 } from 'lucide-react';
import { useAppStore, STAGE_INFO, type GenerationStage } from '@/lib/store';

const STAGE_ICONS: Record<GenerationStage, React.ReactNode> = {
  idle: null,
  image: <ImageIcon size={16} />,
  '3d_multiview': <Box size={16} />,
  '3d_shape': <Box size={16} />,
  '3d_texture': <Palette size={16} />,
  '3d_export': <Download size={16} />,
};

const STAGE_ORDER: GenerationStage[] = ['image', '3d_multiview', '3d_shape', '3d_texture', '3d_export'];

export function GenerationProgress() {
  const isLoading = useAppStore((state) => state.isLoading);
  const generationStage = useAppStore((state) => state.generationStage);
  const generationStartTime = useAppStore((state) => state.generationStartTime);
  const engine3d = useAppStore((state) => state.engine3d);

  const [elapsedTime, setElapsedTime] = useState(0);

  // Update elapsed time every second
  useEffect(() => {
    if (!isLoading || !generationStartTime) {
      setElapsedTime(0);
      return;
    }

    const interval = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - generationStartTime) / 1000));
    }, 1000);

    return () => clearInterval(interval);
  }, [isLoading, generationStartTime]);

  if (!isLoading || generationStage === 'idle') {
    return null;
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return mins > 0 ? `${mins}分${secs}秒` : `${secs}秒`;
  };

  const currentStageIndex = STAGE_ORDER.indexOf(generationStage);
  const stageInfo = STAGE_INFO[generationStage];

  // Determine which stages to show based on engine
  const isMultiviewEngine = ['auto_mv', 'gemini_mv', 'hunyuan3d_mv'].includes(engine3d);
  const visibleStages = isMultiviewEngine
    ? STAGE_ORDER
    : STAGE_ORDER.filter(s => s !== '3d_multiview');

  return (
    <div className="bg-white rounded-xl shadow-card border border-slate-100 p-4 mb-4">
      {/* Header with elapsed time */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Loader2 className="animate-spin text-primary-500" size={18} />
          <span className="font-medium text-slate-700">生成中...</span>
        </div>
        <span className="text-sm text-slate-500 font-mono">{formatTime(elapsedTime)}</span>
      </div>

      {/* Progress steps */}
      <div className="flex items-center gap-1 mb-3">
        {visibleStages.map((stage, index) => {
          const stageIndex = STAGE_ORDER.indexOf(stage);
          const isCompleted = currentStageIndex > stageIndex;
          const isCurrent = stage === generationStage;

          return (
            <div key={stage} className="flex items-center flex-1">
              <div
                className={`flex-1 h-1.5 rounded-full transition-all duration-300 ${
                  isCompleted
                    ? 'bg-green-500'
                    : isCurrent
                    ? 'bg-primary-500 animate-pulse'
                    : 'bg-slate-200'
                }`}
              />
              {index < visibleStages.length - 1 && <div className="w-1" />}
            </div>
          );
        })}
      </div>

      {/* Current stage info */}
      <div className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg">
        <div className={`p-2 rounded-lg ${
          currentStageIndex >= 0 ? 'bg-primary-100 text-primary-600' : 'bg-slate-200 text-slate-500'
        }`}>
          {STAGE_ICONS[generationStage] || <Loader2 size={16} />}
        </div>
        <div className="flex-1">
          <div className="font-medium text-slate-800">{stageInfo.label}</div>
          <div className="text-sm text-slate-500">{stageInfo.description}</div>
        </div>
        {stageInfo.estimatedTime && (
          <div className="text-xs text-slate-400 whitespace-nowrap">
            目安: {stageInfo.estimatedTime}
          </div>
        )}
      </div>

      {/* Stage labels */}
      <div className="flex justify-between mt-2 px-1">
        {visibleStages.map((stage) => {
          const stageIndex = STAGE_ORDER.indexOf(stage);
          const isCompleted = currentStageIndex > stageIndex;
          const isCurrent = stage === generationStage;
          const info = STAGE_INFO[stage];

          return (
            <div
              key={stage}
              className={`text-[10px] text-center transition-colors ${
                isCompleted
                  ? 'text-green-600'
                  : isCurrent
                  ? 'text-primary-600 font-medium'
                  : 'text-slate-400'
              }`}
            >
              {isCompleted ? (
                <CheckCircle2 size={12} className="mx-auto mb-0.5" />
              ) : null}
              <span className="hidden sm:inline">{info.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
