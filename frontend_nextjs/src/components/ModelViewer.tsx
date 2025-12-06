'use client';

import { Suspense, useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, useGLTF } from '@react-three/drei';
import { Box, Download, Loader2, Puzzle } from 'lucide-react';
import { useAppStore } from '@/lib/store';
import { segmentParts, getMeshUrl } from '@/lib/api';

function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url);

  useEffect(() => {
    scene.traverse((child: any) => {
      if (child.isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
      }
    });
  }, [scene]);

  return <primitive object={scene} scale={2} />;
}

function LoadingSpinner() {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100">
      <Loader2 className="animate-spin text-primary-500" size={40} />
    </div>
  );
}

export function ModelViewer() {
  const meshGlbUrl = useAppStore((state) => state.meshGlbUrl);
  const meshObjUrl = useAppStore((state) => state.meshObjUrl);
  const segmentedMeshUrl = useAppStore((state) => state.segmentedMeshUrl);
  const partCount = useAppStore((state) => state.partCount);
  const isSegmenting = useAppStore((state) => state.isSegmenting);
  const setIsSegmenting = useAppStore((state) => state.setIsSegmenting);
  const setSegmentedMesh = useAppStore((state) => state.setSegmentedMesh);
  const setError = useAppStore((state) => state.setError);
  const [isClient, setIsClient] = useState(false);
  const [showSegmented, setShowSegmented] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  // Reset segmented view when new mesh is generated
  useEffect(() => {
    setShowSegmented(false);
  }, [meshGlbUrl]);

  // Handle part segmentation
  const handleSegmentParts = async () => {
    if (!meshGlbUrl) return;

    setIsSegmenting(true);
    setError(null);

    try {
      // Extract just the URL path part (e.g., /api/models/xxx.glb)
      const urlPath = meshGlbUrl.includes('http')
        ? new URL(meshGlbUrl).pathname
        : meshGlbUrl;

      const result = await segmentParts({
        mesh_glb_url: urlPath,
        post_process: true,
        seed: 42,
      });

      if (result.success && result.segmented_mesh_url) {
        setSegmentedMesh(getMeshUrl(result.segmented_mesh_url), result.part_count ?? null);
        setShowSegmented(true);
      } else {
        setError(result.error || 'Part segmentation failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Part segmentation failed');
    } finally {
      setIsSegmenting(false);
    }
  };

  // Determine which mesh to display
  const displayMeshUrl = showSegmented && segmentedMeshUrl ? segmentedMeshUrl : meshGlbUrl;

  return (
    <div className="bg-white rounded-2xl shadow-card border border-slate-100 p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-slate-900 flex items-center gap-2">
          <Box className="text-primary-500" size={20} />
          3Dモデル
          {showSegmented && partCount && (
            <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-full">
              {partCount}パーツ
            </span>
          )}
        </h2>

        {/* View Toggle (when segmented mesh is available) */}
        {segmentedMeshUrl && (
          <div className="flex bg-slate-100 p-1 rounded-lg">
            <button
              onClick={() => setShowSegmented(false)}
              className={`px-3 py-1 text-xs rounded-md transition-all ${
                !showSegmented
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              オリジナル
            </button>
            <button
              onClick={() => setShowSegmented(true)}
              className={`px-3 py-1 text-xs rounded-md transition-all ${
                showSegmented
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              分割済み
            </button>
          </div>
        )}
      </div>

      {/* 3D Viewer */}
      <div className="aspect-square relative bg-gradient-to-br from-slate-50 to-slate-100 border border-slate-200 rounded-xl overflow-hidden">
        {displayMeshUrl && isClient ? (
          <Suspense fallback={<LoadingSpinner />}>
            <Canvas
              camera={{ position: [0, 0, 4], fov: 50 }}
              className="w-full h-full"
            >
              <ambientLight intensity={0.6} />
              <spotLight
                position={[10, 10, 10]}
                angle={0.15}
                penumbra={1}
                intensity={1}
                castShadow
              />
              <pointLight position={[-10, -10, -10]} intensity={0.5} />
              <Model url={displayMeshUrl} />
              <OrbitControls
                autoRotate
                autoRotateSpeed={2}
                enablePan={true}
                enableZoom={true}
              />
              <Environment preset="studio" />
            </Canvas>
          </Suspense>
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400">
            <div className="p-4 bg-slate-200 rounded-full mb-3">
              <Box size={32} />
            </div>
            <p className="font-medium">3Dモデルがまだ生成されていません</p>
            <p className="text-sm text-slate-400 mt-1">生成後に表示されます</p>
          </div>
        )}

        {/* Segmenting Overlay */}
        {isSegmenting && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/80 backdrop-blur-sm">
            <Loader2 className="animate-spin text-purple-500 mb-2" size={40} />
            <p className="text-slate-700 font-medium">パーツ分割中...</p>
            <p className="text-sm text-slate-500">P3-SAMで処理しています</p>
          </div>
        )}
      </div>

      {/* Download Buttons */}
      <div className="flex gap-3 mt-4">
        <a
          href={meshGlbUrl || '#'}
          download="model.glb"
          className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl border text-sm font-medium transition-all ${
            meshGlbUrl
              ? 'bg-slate-50 border-slate-200 text-slate-700 hover:bg-primary-50 hover:border-primary-300 hover:text-primary-700'
              : 'bg-slate-100 border-slate-200 text-slate-400 cursor-not-allowed'
          }`}
          onClick={(e) => !meshGlbUrl && e.preventDefault()}
        >
          <Download size={16} />
          GLBをダウンロード
        </a>
        <a
          href={meshObjUrl || '#'}
          download="model.obj"
          className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl border text-sm font-medium transition-all ${
            meshObjUrl
              ? 'bg-slate-50 border-slate-200 text-slate-700 hover:bg-primary-50 hover:border-primary-300 hover:text-primary-700'
              : 'bg-slate-100 border-slate-200 text-slate-400 cursor-not-allowed'
          }`}
          onClick={(e) => !meshObjUrl && e.preventDefault()}
        >
          <Download size={16} />
          OBJをダウンロード
        </a>
      </div>

      {/* Part Segmentation Button */}
      <button
        onClick={handleSegmentParts}
        disabled={!meshGlbUrl || isSegmenting}
        className={`w-full flex items-center justify-center gap-2 py-2.5 mt-3 rounded-xl border text-sm font-medium transition-all ${
          meshGlbUrl && !isSegmenting
            ? 'bg-purple-50 border-purple-200 text-purple-700 hover:bg-purple-100 hover:border-purple-300'
            : 'bg-slate-100 border-slate-200 text-slate-400 cursor-not-allowed'
        }`}
      >
        {isSegmenting ? (
          <>
            <Loader2 size={16} className="animate-spin" />
            パーツ分割中...
          </>
        ) : (
          <>
            <Puzzle size={16} />
            パーツ分割 (P3-SAM)
          </>
        )}
      </button>

      {/* Segmented Mesh Download */}
      {segmentedMeshUrl && (
        <a
          href={segmentedMeshUrl}
          download="model_segmented.glb"
          className="w-full flex items-center justify-center gap-2 py-2.5 mt-2 rounded-xl border text-sm font-medium transition-all bg-purple-50 border-purple-200 text-purple-700 hover:bg-purple-100 hover:border-purple-300"
        >
          <Download size={16} />
          分割済みGLBをダウンロード
        </a>
      )}
    </div>
  );
}
