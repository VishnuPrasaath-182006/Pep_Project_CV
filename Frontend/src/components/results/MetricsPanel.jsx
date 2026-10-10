import React from 'react';
import { BarChart3, Clock, Cpu, Droplets, Percent, CheckCircle2 } from 'lucide-react';

export default function MetricsPanel({ result }) {
  if (!result) return null;

  return (
    <div className="card" style={{ marginBottom: '1.25rem' }}>
      <div className="card-header">
        <h3 className="card-title">
          <BarChart3 size={18} color="#2563eb" />
          <span>Quantitative Flood Telemetry</span>
        </h3>
        <span style={{ fontSize: '0.75rem', fontFamily: 'JetBrains Mono, monospace', color: '#059669', display: 'flex', alignItems: 'center', gap: '3px' }}>
          <CheckCircle2 size={13} /> Genuine Prediction
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '0.75rem' }}>
        {/* Coverage Percentage */}
        <div style={{
          padding: '0.85rem',
          backgroundColor: '#eff6ff',
          borderRadius: '4px',
          border: '1px solid #bfdbfe'
        }}>
          <div style={{ fontSize: '0.7rem', fontWeight: 700, color: '#1e40af', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Percent size={12} />
            <span>Flood Coverage</span>
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#1e3a8a', fontFamily: 'JetBrains Mono, monospace', marginTop: '0.25rem' }}>
            {result.flood_coverage_percentage}%
          </div>
          <div style={{ fontSize: '0.7rem', color: '#3b82f6', marginTop: '2px' }}>
            of total valid analyzed pixels
          </div>
        </div>

        {/* Flood Pixel Count */}
        <div style={{
          padding: '0.85rem',
          backgroundColor: '#f8fafc',
          borderRadius: '4px',
          border: '1px solid #e2e8f0'
        }}>
          <div style={{ fontSize: '0.7rem', fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Droplets size={12} color="#0ea5e9" />
            <span>Inundated Pixels</span>
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#0f172a', fontFamily: 'JetBrains Mono, monospace', marginTop: '0.25rem' }}>
            {result.flood_pixels.toLocaleString()}
          </div>
          <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '2px' }}>
            / {result.valid_pixels.toLocaleString()} px
          </div>
        </div>

        {/* Processing Time */}
        <div style={{
          padding: '0.85rem',
          backgroundColor: '#f8fafc',
          borderRadius: '4px',
          border: '1px solid #e2e8f0'
        }}>
          <div style={{ fontSize: '0.7rem', fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Clock size={12} color="#f59e0b" />
            <span>Inference Latency</span>
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#0f172a', fontFamily: 'JetBrains Mono, monospace', marginTop: '0.25rem' }}>
            {result.processing_time_ms} ms
          </div>
          <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '2px' }}>
            Tiled forward execution
          </div>
        </div>

        {/* Hardware Execution Engine */}
        <div style={{
          padding: '0.85rem',
          backgroundColor: '#f8fafc',
          borderRadius: '4px',
          border: '1px solid #e2e8f0'
        }}>
          <div style={{ fontSize: '0.7rem', fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Cpu size={12} color="#10b981" />
            <span>Compute Device</span>
          </div>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0f172a', fontFamily: 'JetBrains Mono, monospace', marginTop: '0.25rem' }}>
            {result.device}
          </div>
          <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '2px' }}>
            Mean Conf: {(result.mean_confidence * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  );
}
