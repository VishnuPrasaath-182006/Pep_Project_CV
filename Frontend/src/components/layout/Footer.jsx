import React from 'react';
import { ShieldCheck, HardDrive, Cpu } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="footer">
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <ShieldCheck size={16} color="#0f766e" />
        <span>CAVE-Flood Intelligent Earth Observation Platform &copy; 2026. Cryptographically Verified Checkpoints.</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.75rem' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <Cpu size={14} /> PyTorch 2.x Selective Scan SS2D
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <HardDrive size={14} /> Checkpoint SHA-256: 58674c...
        </span>
      </div>
    </footer>
  );
}
