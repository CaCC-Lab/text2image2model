'use client';

import { Loader2, Sparkles } from 'lucide-react';
import { useAppStore } from '@/lib/store';
import { generateImageOnly, generate3DOnly, getMeshUrl } from '@/lib/api';
import toast from 'react-hot-toast';

export function GenerateButton() {
  const prompt = useAppStore((state) => state.prompt);
  const checkpoint = useAppStore((state) => state.checkpoint);
  const removeBackground = useAppStore((state) => state.removeBackground);
  const foregroundRatio = useAppStore((state) => state.foregroundRatio);
  const mcResolution = useAppStore((state) => state.mcResolution);
  const engine3d = useAppStore((state) => state.engine3d);
  const imageEngine = useAppStore((state) => state.imageEngine);
  const meshQuality = useAppStore((state) => state.meshQuality);
  const isLoading = useAppStore((state) => state.isLoading);
  const generationStage = useAppStore((state) => state.generationStage);
  const backendConnected = useAppStore((state) => state.backendConnected);
  const setIsLoading = useAppStore((state) => state.setIsLoading);
  const setGenerationStage = useAppStore((state) => state.setGenerationStage);
  const setError = useAppStore((state) => state.setError);
  const setGeneratedImage = useAppStore((state) => state.setGeneratedImage);
  const setGenerationResult = useAppStore((state) => state.setGenerationResult);
  const setMultiviewImages = useAppStore((state) => state.setMultiviewImages);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      toast.error('プロンプトを入力してください');
      return;
    }

    if (!backendConnected) {
      toast.error('バックエンドに接続されていません');
      return;
    }

    setIsLoading(true);
    setError(null);
    setGeneratedImage(null);

    let imageBase64: string | null = null;
    let imageTime = 0;

    try {
      // Stage 1: Generate image
      setGenerationStage('image');
      toast('画像を生成中...', { icon: '🎨' });

      const imageResult = await generateImageOnly({
        prompt,
        checkpoint,
        image_engine: imageEngine,
      });

      if (!imageResult.success) {
        throw new Error(imageResult.error || '画像生成に失敗しました');
      }

      imageBase64 = imageResult.image;
      imageTime = imageResult.processing_time;

      // Display image immediately
      setGeneratedImage(imageBase64);
      toast.success(`画像生成完了！ (${imageTime.toFixed(1)}秒)`);

      // Stage 2: Generate 3D model
      setGenerationStage('3d');
      toast('3Dモデルを生成中...', { icon: '🧊' });

      const threeDResult = await generate3DOnly({
        image: imageBase64,
        remove_background: removeBackground,
        foreground_ratio: foregroundRatio,
        mc_resolution: mcResolution,
        engine_3d: engine3d,
        mesh_quality: meshQuality,
      });

      if (!threeDResult.success) {
        throw new Error(threeDResult.error || '3D生成に失敗しました');
      }

      const totalTime = imageTime + threeDResult.processing_time;

      setGenerationResult({
        prompt,
        generatedImage: imageBase64,
        processedImage: threeDResult.processed_image,
        meshObjUrl: getMeshUrl(threeDResult.mesh_obj_url),
        meshGlbUrl: getMeshUrl(threeDResult.mesh_glb_url),
        processingTime: totalTime,
        timestamp: Date.now(),
        engine3d: threeDResult.engine_3d,
      });

      // Set multiview images if available (from auto_mv engine)
      if (threeDResult.multiview_front || threeDResult.multiview_left || threeDResult.multiview_right || threeDResult.multiview_back) {
        setMultiviewImages(
          threeDResult.multiview_front || null,
          threeDResult.multiview_left || null,
          threeDResult.multiview_right || null,
          threeDResult.multiview_back || null
        );
      }

      toast.success(`全処理完了！ (${totalTime.toFixed(1)}秒)`);
    } catch (error) {
      const message = error instanceof Error ? error.message : '生成に失敗しました';
      setError(message);
      toast.error(message);
    } finally {
      setIsLoading(false);
      setGenerationStage('idle');
    }
  };

  const getButtonText = () => {
    if (!isLoading) return 'テキストから生成';
    if (generationStage === 'image') return '画像生成中...';
    if (generationStage === '3d') return '3Dモデル生成中...';
    return '生成中...';
  };

  return (
    <button
      onClick={handleGenerate}
      disabled={isLoading || !backendConnected}
      className="w-full bg-gradient-to-r from-primary-500 to-primary-600 text-white font-semibold py-4 rounded-xl
                 hover:from-primary-600 hover:to-primary-700 transition-all shadow-lg shadow-primary-500/25
                 disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none
                 flex items-center justify-center gap-2 active:scale-[0.99]"
    >
      {isLoading ? (
        <>
          <Loader2 className="animate-spin" size={20} />
          {getButtonText()}
        </>
      ) : (
        <>
          <Sparkles size={20} />
          {getButtonText()}
        </>
      )}
    </button>
  );
}
