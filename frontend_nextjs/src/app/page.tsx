'use client';

import { useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Header,
  PromptInput,
  ImageUpload,
  SettingsPanel,
  GenerateButton,
  GenerationProgress,
  ImageDisplay,
  ModelViewer,
  StatusBar,
  Footer,
} from '@/components';
import { checkHealth } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import { Wand2, Upload } from 'lucide-react';

export default function Home() {
  const setBackendConnected = useAppStore((state) => state.setBackendConnected);
  const inputMode = useAppStore((state) => state.inputMode);
  const setInputMode = useAppStore((state) => state.setInputMode);

  useEffect(() => {
    const checkBackend = async () => {
      const connected = await checkHealth();
      setBackendConnected(connected);
    };

    checkBackend();
    const interval = setInterval(checkBackend, 10000);
    return () => clearInterval(interval);
  }, [setBackendConnected]);

  return (
    <div className="min-h-screen">
      <Header />

      <main className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-10"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4">
            AI 3D Generator
          </h1>
          <p className="text-slate-600 text-lg max-w-2xl mx-auto">
            テキストや画像から高品質な3Dモデルを生成します。
            SDXL-Lightning & Hunyuan3D-2 搭載。
          </p>
        </motion.div>

        {/* Input Mode Tabs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="flex justify-center mb-8"
        >
          <div className="inline-flex bg-slate-100 p-1.5 rounded-xl gap-1">
            <button
              onClick={() => setInputMode('text')}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-medium transition-all ${
                inputMode === 'text'
                  ? 'bg-white text-slate-900 shadow-md'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Wand2 size={18} />
              テキストから生成
            </button>
            <button
              onClick={() => setInputMode('image')}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-medium transition-all ${
                inputMode === 'image'
                  ? 'bg-white text-slate-900 shadow-md'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Upload size={18} />
              画像から生成
            </button>
          </div>
        </motion.div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Input */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="space-y-6"
          >
            {inputMode === 'text' ? (
              <>
                <PromptInput />
                <GenerateButton />
                <GenerationProgress />
              </>
            ) : (
              <ImageUpload />
            )}

            <SettingsPanel />
            <StatusBar />
          </motion.div>

          {/* Right Column - Output */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="space-y-6"
          >
            <ImageDisplay />
            <ModelViewer />
          </motion.div>
        </div>

        <Footer />
      </main>
    </div>
  );
}
