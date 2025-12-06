'use client';

import { Suspense, useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, useGLTF } from '@react-three/drei';
import { Box, Download, Loader2 } from 'lucide-react';
import { useAppStore } from '@/lib/store';

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
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  return (
    <div className="bg-white rounded-2xl shadow-card border border-slate-100 p-6">
      <h2 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
        <Box className="text-primary-500" size={20} />
        3Dモデル
      </h2>

      {/* 3D Viewer */}
      <div className="aspect-square relative bg-gradient-to-br from-slate-50 to-slate-100 border border-slate-200 rounded-xl overflow-hidden">
        {meshGlbUrl && isClient ? (
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
              <Model url={meshGlbUrl} />
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
    </div>
  );
}
