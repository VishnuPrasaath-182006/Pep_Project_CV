import React from 'react';
import { Loader2 } from 'lucide-react';

export default function LoadingSpinner({ message = 'Executing inference...', submessage = '' }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '3rem 2rem',
      backgroundColor: '#ffffff',
      borderRadius: '4px',
      border: '1px solid #e2e8f0',
      textAlign: 'center',
      gap: '0.75rem'
    }}>
      <Loader2 size={36} color="#2563eb" style={{ animation: 'spin 1s linear infinite' }} />
      <div style={{ fontWeight: 600, fontSize: '0.95rem', color: '#102a43' }}>
        {message}
      </div>
      {submessage && (
        <div style={{ fontSize: '0.8rem', color: '#64748b', maxWidth: '420px', fontFamily: 'JetBrains Mono, monospace' }}>
          {submessage}
        </div>
      )}
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
