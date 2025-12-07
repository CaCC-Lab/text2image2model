'use client';

import { useState, useEffect } from 'react';
import { ImageIcon, Check, Layers } from 'lucide-react';
import Image from 'next/image';
import { useAppStore } from '@/lib/store';

type TabType = 'generated' | 'processed' | 'multiview';

export function ImageDisplay() {
  const [activeTab, setActiveTab] = useState<TabType>('generated');
  const generatedImage = useAppStore((state) => state.generatedImage);
  const processedImage = useAppStore((state) => state.processedImage);
  const multiviewFront = useAppStore((state) => state.multiviewFront);
  const multiviewLeft = useAppStore((state) => state.multiviewLeft);
  const multiviewRight = useAppStore((state) => state.multiviewRight);
  const multiviewBack = useAppStore((state) => state.multiviewBack);

  const hasMultiview = multiviewFront || multiviewLeft || multiviewRight || multiviewBack;

  // Auto-switch to multiview tab when multiview images become available
  useEffect(() => {
    if (hasMultiview) {
      setActiveTab('multiview');
    } else if (processedImage) {
      setActiveTab('processed');
    }
  }, [processedImage, hasMultiview]);

  // Reset to generated tab when a new generation starts (generatedImage changes without processedImage)
  useEffect(() => {
    if (generatedImage && !processedImage && !hasMultiview) {
      setActiveTab('generated');
    }
  }, [generatedImage, processedImage, hasMultiview]);

  const currentImage = activeTab === 'generated' ? generatedImage : processedImage;

  return (
    <div className="bg-white rounded-2xl shadow-card border border-slate-100 p-6">
      <h2 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
        <ImageIcon className="text-primary-500" size={20} />
        生成された画像
      </h2>

      {/* Tabs */}
      <div className="flex gap-2 mb-4 flex-wrap">
        <button
          onClick={() => setActiveTab('generated')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            activeTab === 'generated'
              ? 'bg-primary-500 text-white shadow-md shadow-primary-500/25'
              : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
          }`}
        >
          生成画像
        </button>
        <button
          onClick={() => setActiveTab('processed')}
          className={`relative px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            activeTab === 'processed'
              ? 'bg-primary-500 text-white shadow-md shadow-primary-500/25'
              : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
          }`}
        >
          処理済み画像
          {processedImage && activeTab !== 'processed' && (
            <span className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full flex items-center justify-center">
              <Check size={10} className="text-white" />
            </span>
          )}
        </button>
        {hasMultiview && (
          <button
            onClick={() => setActiveTab('multiview')}
            className={`relative px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-1 ${
              activeTab === 'multiview'
                ? 'bg-primary-500 text-white shadow-md shadow-primary-500/25'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            <Layers size={14} />
            マルチビュー
            {activeTab !== 'multiview' && (
              <span className="absolute -top-1 -right-1 w-4 h-4 bg-purple-500 rounded-full flex items-center justify-center">
                <Check size={10} className="text-white" />
              </span>
            )}
          </button>
        )}
      </div>

      {/* Image Display */}
      {activeTab === 'multiview' && hasMultiview ? (
        <div className="grid grid-cols-2 gap-2">
          {/* Front View */}
          <div className="aspect-square relative bg-gradient-to-br from-slate-50 to-slate-100 border border-slate-200 rounded-xl overflow-hidden">
            {multiviewFront ? (
              <>
                <Image
                  src={`data:image/png;base64,${multiviewFront}`}
                  alt="正面"
                  fill
                  className="object-contain"
                />
                <div className="absolute bottom-2 left-2 bg-black/50 text-white text-xs px-2 py-1 rounded">
                  正面
                </div>
              </>
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-slate-400 text-sm">
                正面なし
              </div>
            )}
          </div>
          {/* Right View */}
          <div className="aspect-square relative bg-gradient-to-br from-slate-50 to-slate-100 border border-slate-200 rounded-xl overflow-hidden">
            {multiviewRight ? (
              <>
                <Image
                  src={`data:image/png;base64,${multiviewRight}`}
                  alt="右側面"
                  fill
                  className="object-contain"
                />
                <div className="absolute bottom-2 left-2 bg-black/50 text-white text-xs px-2 py-1 rounded">
                  右側面
                </div>
              </>
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-slate-400 text-sm">
                右側面なし
              </div>
            )}
          </div>
          {/* Left View */}
          <div className="aspect-square relative bg-gradient-to-br from-slate-50 to-slate-100 border border-slate-200 rounded-xl overflow-hidden">
            {multiviewLeft ? (
              <>
                <Image
                  src={`data:image/png;base64,${multiviewLeft}`}
                  alt="左側面"
                  fill
                  className="object-contain"
                />
                <div className="absolute bottom-2 left-2 bg-black/50 text-white text-xs px-2 py-1 rounded">
                  左側面
                </div>
              </>
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-slate-400 text-sm">
                左側面なし
              </div>
            )}
          </div>
          {/* Back View */}
          <div className="aspect-square relative bg-gradient-to-br from-slate-50 to-slate-100 border border-slate-200 rounded-xl overflow-hidden">
            {multiviewBack ? (
              <>
                <Image
                  src={`data:image/png;base64,${multiviewBack}`}
                  alt="背面"
                  fill
                  className="object-contain"
                />
                <div className="absolute bottom-2 left-2 bg-black/50 text-white text-xs px-2 py-1 rounded">
                  背面
                </div>
              </>
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-slate-400 text-sm">
                背面なし
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="aspect-square relative bg-gradient-to-br from-slate-50 to-slate-100 border border-slate-200 rounded-xl overflow-hidden">
          {currentImage ? (
            <Image
              src={`data:image/png;base64,${currentImage}`}
              alt={activeTab === 'generated' ? '生成された画像' : '処理済み画像'}
              fill
              className="object-contain"
            />
          ) : (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400">
              <div className="p-4 bg-slate-200 rounded-full mb-3">
                <ImageIcon size={32} />
              </div>
              <p className="font-medium">画像がまだ生成されていません</p>
              <p className="text-sm text-slate-400 mt-1">プロンプトを入力して生成ボタンをクリック</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
