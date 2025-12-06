import { Sparkles } from 'lucide-react';

export function Footer() {
  return (
    <footer className="border-t border-slate-200 py-6 text-center mt-12">
      <div className="flex items-center justify-center gap-2 text-slate-600">
        <Sparkles size={16} className="text-primary-500" />
        <span>SDXL-Lightning & Hunyuan3D-2 搭載</span>
      </div>
      <p className="text-slate-400 text-sm mt-2">
        AI 3D Generator v2.0 - Built with Next.js
      </p>
    </footer>
  );
}
