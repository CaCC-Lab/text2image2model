'use client';

import { useState } from 'react';
import { ChevronDown, ChevronUp, Settings, Zap, Star } from 'lucide-react';
import { useAppStore } from '@/lib/store';
import { CHECKPOINT_INFO, ENGINE_3D_INFO, IMAGE_ENGINE_INFO, MESH_QUALITY_INFO, type CheckpointType, type Engine3DType, type ImageEngineType, type MeshQualityType } from '@/types/api';

export function SettingsPanel() {
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const checkpoint = useAppStore((state) => state.checkpoint);
  const setCheckpoint = useAppStore((state) => state.setCheckpoint);
  const removeBackground = useAppStore((state) => state.removeBackground);
  const setRemoveBackground = useAppStore((state) => state.setRemoveBackground);
  const foregroundRatio = useAppStore((state) => state.foregroundRatio);
  const setForegroundRatio = useAppStore((state) => state.setForegroundRatio);
  const mcResolution = useAppStore((state) => state.mcResolution);
  const setMcResolution = useAppStore((state) => state.setMcResolution);
  const engine3d = useAppStore((state) => state.engine3d);
  const setEngine3d = useAppStore((state) => state.setEngine3d);
  const imageEngine = useAppStore((state) => state.imageEngine);
  const setImageEngine = useAppStore((state) => state.setImageEngine);
  const meshQuality = useAppStore((state) => state.meshQuality);
  const setMeshQuality = useAppStore((state) => state.setMeshQuality);
  const inputMode = useAppStore((state) => state.inputMode);

  const checkpoints = Object.keys(CHECKPOINT_INFO) as CheckpointType[];
  // Filter engines based on input mode:
  // - Text mode: hide hunyuan3d_mv and gemini_mv (require image upload)
  // - Image mode: hide auto_mv (for text-to-image pipeline)
  // - Always hide: hunyuan_api (disabled for now, implementation kept)
  const allEngines = Object.keys(ENGINE_3D_INFO) as Engine3DType[];
  const hiddenEngines = ['hunyuan_api']; // Disabled engines (implementation kept in code)
  const engines = inputMode === 'text'
    ? allEngines.filter(e => e !== 'hunyuan3d_mv' && e !== 'gemini_mv' && !hiddenEngines.includes(e))
    : allEngines.filter(e => e !== 'auto_mv' && !hiddenEngines.includes(e));
  const imageEngines = Object.keys(IMAGE_ENGINE_INFO) as ImageEngineType[];
  const meshQualities = Object.keys(MESH_QUALITY_INFO) as MeshQualityType[];

  const renderRating = (count: number, max: number, icon: 'speed' | 'quality') => {
    return (
      <div className="flex gap-0.5">
        {Array.from({ length: max }).map((_, i) => (
          <span
            key={i}
            className={`text-xs ${
              i < count
                ? icon === 'speed' ? 'text-amber-500' : 'text-primary-500'
                : 'text-slate-300'
            }`}
          >
            {icon === 'speed' ? '⚡' : '★'}
          </span>
        ))}
      </div>
    );
  };

  return (
    <div className="bg-white rounded-2xl shadow-card border border-slate-100 p-6 space-y-5">
      <h2 className="text-lg font-semibold text-slate-900 flex items-center gap-2">
        <Settings className="text-primary-500" size={20} />
        生成設定
      </h2>

      {/* Checkpoint Selection */}
      <div>
        <label className="block text-slate-600 text-sm font-medium mb-2">画像品質</label>
        <select
          value={checkpoint}
          onChange={(e) => setCheckpoint(e.target.value as CheckpointType)}
          className="w-full bg-slate-50 border border-slate-200 rounded-xl p-3 text-slate-900
                     focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 outline-none"
        >
          {checkpoints.map((cp) => (
            <option key={cp} value={cp}>
              {CHECKPOINT_INFO[cp].name}
            </option>
          ))}
        </select>
        <p className="text-slate-500 text-xs mt-2">{CHECKPOINT_INFO[checkpoint].description}</p>
        <div className="flex gap-6 mt-2">
          <div className="flex items-center gap-2">
            <span className="text-slate-500 text-xs">速度</span>
            {renderRating(CHECKPOINT_INFO[checkpoint].speed, 5, 'speed')}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-slate-500 text-xs">品質</span>
            {renderRating(CHECKPOINT_INFO[checkpoint].quality, 4, 'quality')}
          </div>
        </div>
      </div>

      {/* Image Engine Selection */}
      <div>
        <label className="block text-slate-600 text-sm font-medium mb-2">画像生成エンジン</label>
        <select
          value={imageEngine}
          onChange={(e) => setImageEngine(e.target.value as ImageEngineType)}
          className="w-full bg-slate-50 border border-slate-200 rounded-xl p-3 text-slate-900
                     focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 outline-none"
        >
          {imageEngines.map((eng) => (
            <option key={eng} value={eng}>
              {IMAGE_ENGINE_INFO[eng].name}
            </option>
          ))}
        </select>
        <p className="text-slate-500 text-xs mt-2">{IMAGE_ENGINE_INFO[imageEngine].description}</p>
        <div className="flex gap-6 mt-2">
          <div className="flex items-center gap-2">
            <span className="text-slate-500 text-xs">速度</span>
            {renderRating(IMAGE_ENGINE_INFO[imageEngine].speed, 5, 'speed')}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-slate-500 text-xs">品質</span>
            {renderRating(IMAGE_ENGINE_INFO[imageEngine].quality, 5, 'quality')}
          </div>
        </div>
      </div>

      {/* 3D Engine Selection */}
      <div>
        <label className="block text-slate-600 text-sm font-medium mb-2">3Dエンジン</label>
        <select
          value={engine3d}
          onChange={(e) => setEngine3d(e.target.value as Engine3DType)}
          className="w-full bg-slate-50 border border-slate-200 rounded-xl p-3 text-slate-900
                     focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 outline-none"
        >
          {engines.map((eng) => (
            <option key={eng} value={eng}>
              {ENGINE_3D_INFO[eng].name}
            </option>
          ))}
        </select>
        <p className="text-slate-500 text-xs mt-2">{ENGINE_3D_INFO[engine3d].description}</p>
        <div className="flex gap-6 mt-2">
          <div className="flex items-center gap-2">
            <span className="text-slate-500 text-xs">速度</span>
            {renderRating(ENGINE_3D_INFO[engine3d].speed, 5, 'speed')}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-slate-500 text-xs">品質</span>
            {renderRating(ENGINE_3D_INFO[engine3d].quality, 5, 'quality')}
          </div>
        </div>
      </div>

      {/* Mesh Quality Selection - Only show when Hunyuan3D is selected */}
      {engine3d === 'hunyuan3d' && (
        <div>
          <label className="block text-slate-600 text-sm font-medium mb-2">メッシュ品質</label>
          <select
            value={meshQuality}
            onChange={(e) => setMeshQuality(e.target.value as MeshQualityType)}
            className="w-full bg-slate-50 border border-slate-200 rounded-xl p-3 text-slate-900
                       focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 outline-none"
          >
            {meshQualities.map((mq) => (
              <option key={mq} value={mq}>
                {MESH_QUALITY_INFO[mq].name}
              </option>
            ))}
          </select>
          <p className="text-slate-500 text-xs mt-2">{MESH_QUALITY_INFO[meshQuality].description}</p>
          <div className="flex gap-6 mt-2">
            <div className="flex items-center gap-2">
              <span className="text-slate-500 text-xs">速度</span>
              {renderRating(MESH_QUALITY_INFO[meshQuality].speed, 5, 'speed')}
            </div>
            <div className="flex items-center gap-2">
              <span className="text-slate-500 text-xs">品質</span>
              {renderRating(MESH_QUALITY_INFO[meshQuality].quality, 5, 'quality')}
            </div>
          </div>
        </div>
      )}

      {/* Remove Background Toggle */}
      <div className="flex items-center justify-between py-2">
        <div>
          <p className="text-slate-700 font-medium">背景を削除</p>
          <p className="text-slate-500 text-xs">3D変換の精度向上のため背景を自動削除</p>
        </div>
        <button
          onClick={() => setRemoveBackground(!removeBackground)}
          className={`relative w-12 h-6 rounded-full transition-colors ${
            removeBackground ? 'bg-primary-500' : 'bg-slate-300'
          }`}
        >
          <div
            className={`absolute top-1 w-4 h-4 bg-white rounded-full shadow transition-transform ${
              removeBackground ? 'translate-x-7' : 'translate-x-1'
            }`}
          />
        </button>
      </div>

      {/* Advanced Settings */}
      <div className="border border-slate-200 rounded-xl overflow-hidden">
        <button
          onClick={() => setAdvancedOpen(!advancedOpen)}
          className="w-full flex items-center justify-between p-4 text-slate-700 hover:bg-slate-50 transition-colors"
        >
          <span className="font-medium">詳細設定</span>
          {advancedOpen ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </button>
        {advancedOpen && (
          <div className="p-4 space-y-4 border-t border-slate-200 bg-slate-50">
            {/* Foreground Ratio */}
            <div>
              <div className="flex justify-between mb-2">
                <label className="text-slate-600 text-sm">前景比率</label>
                <span className="text-slate-900 text-sm font-medium">{foregroundRatio.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="1"
                step="0.05"
                value={foregroundRatio}
                onChange={(e) => setForegroundRatio(parseFloat(e.target.value))}
                className="w-full accent-primary-500"
              />
            </div>

            {/* MC Resolution */}
            <div>
              <div className="flex justify-between mb-2">
                <label className="text-slate-600 text-sm">3D解像度</label>
                <span className="text-slate-900 text-sm font-medium">{mcResolution}</span>
              </div>
              <input
                type="range"
                min="32"
                max="256"
                step="32"
                value={mcResolution}
                onChange={(e) => setMcResolution(parseInt(e.target.value))}
                className="w-full accent-primary-500"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
