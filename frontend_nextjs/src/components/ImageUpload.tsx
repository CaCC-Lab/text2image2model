'use client';

import { useState, useCallback, useRef } from 'react';
import { Upload, X, Image as ImageIcon, Loader2 } from 'lucide-react';
import Image from 'next/image';
import { useAppStore } from '@/lib/store';
import { generate3DOnly, getMeshUrl } from '@/lib/api';
import toast from 'react-hot-toast';

export function ImageUpload() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const removeBackground = useAppStore((state) => state.removeBackground);
  const foregroundRatio = useAppStore((state) => state.foregroundRatio);
  const mcResolution = useAppStore((state) => state.mcResolution);
  const engine3d = useAppStore((state) => state.engine3d);
  const meshQuality = useAppStore((state) => state.meshQuality);
  const backendConnected = useAppStore((state) => state.backendConnected);
  const setGenerationResult = useAppStore((state) => state.setGenerationResult);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      toast.error('画像ファイルを選択してください');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      // Extract base64 data (remove data:image/...;base64, prefix)
      const base64 = result.split(',')[1];
      setUploadedImage(base64);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const clearImage = () => {
    setUploadedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleGenerate3D = async () => {
    if (!uploadedImage) {
      toast.error('画像をアップロードしてください');
      return;
    }

    if (!backendConnected) {
      toast.error('バックエンドに接続されていません');
      return;
    }

    setIsGenerating(true);

    try {
      toast('3Dモデルを生成中...', { icon: '🧊' });

      const result = await generate3DOnly({
        image: uploadedImage,
        remove_background: removeBackground,
        foreground_ratio: foregroundRatio,
        mc_resolution: mcResolution,
        engine_3d: engine3d,
        mesh_quality: meshQuality,
      });

      if (!result.success) {
        throw new Error(result.error || '3D生成に失敗しました');
      }

      setGenerationResult({
        prompt: '[画像からアップロード]',
        generatedImage: uploadedImage,
        processedImage: result.processed_image,
        meshObjUrl: getMeshUrl(result.mesh_obj_url),
        meshGlbUrl: getMeshUrl(result.mesh_glb_url),
        processingTime: result.processing_time,
        timestamp: Date.now(),
        engine3d: result.engine_3d,
      });

      toast.success(`3Dモデル生成完了！ (${result.processing_time.toFixed(1)}秒)`);
    } catch (error) {
      const message = error instanceof Error ? error.message : '生成に失敗しました';
      toast.error(message);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-card border border-slate-100 p-6">
      <h2 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
        <Upload className="text-primary-500" size={20} />
        画像から3Dモデル生成
      </h2>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileInput}
        className="hidden"
      />

      {uploadedImage ? (
        <div className="space-y-4">
          <div className="relative aspect-square rounded-xl overflow-hidden bg-slate-50 border border-slate-200">
            <Image
              src={`data:image/png;base64,${uploadedImage}`}
              alt="アップロードされた画像"
              fill
              className="object-contain"
            />
            <button
              onClick={clearImage}
              className="absolute top-2 right-2 p-2 bg-white/90 hover:bg-white rounded-full shadow-lg transition-all"
            >
              <X size={16} className="text-slate-600" />
            </button>
          </div>

          <button
            onClick={handleGenerate3D}
            disabled={isGenerating || !backendConnected}
            className="w-full bg-gradient-to-r from-accent-violet to-accent-pink text-white font-semibold py-3 px-6 rounded-xl
                       hover:opacity-90 transition-all shadow-button disabled:opacity-50 disabled:cursor-not-allowed
                       flex items-center justify-center gap-2"
          >
            {isGenerating ? (
              <>
                <Loader2 className="animate-spin" size={20} />
                3Dモデル生成中...
              </>
            ) : (
              <>
                <ImageIcon size={20} />
                この画像から3Dモデルを生成
              </>
            )}
          </button>
        </div>
      ) : (
        <div
          onClick={handleClick}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`aspect-video rounded-xl border-2 border-dashed transition-all cursor-pointer
                      flex flex-col items-center justify-center gap-3
                      ${isDragging
                        ? 'border-primary-500 bg-primary-50'
                        : 'border-slate-300 bg-slate-50 hover:border-primary-400 hover:bg-slate-100'
                      }`}
        >
          <div className={`p-4 rounded-full ${isDragging ? 'bg-primary-100' : 'bg-slate-200'}`}>
            <Upload size={24} className={isDragging ? 'text-primary-600' : 'text-slate-500'} />
          </div>
          <div className="text-center">
            <p className="text-slate-700 font-medium">
              画像をドラッグ＆ドロップ
            </p>
            <p className="text-slate-500 text-sm">
              または クリックしてファイルを選択
            </p>
          </div>
          <p className="text-slate-400 text-xs">
            PNG, JPG, WEBP に対応
          </p>
        </div>
      )}
    </div>
  );
}
