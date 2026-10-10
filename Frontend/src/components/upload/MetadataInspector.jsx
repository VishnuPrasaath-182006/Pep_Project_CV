import React from 'react';
import { Info, Globe, Layers, CheckCircle2, AlertTriangle, Compass } from 'lucide-react';
import Badge from '../common/Badge';

export default function MetadataInspector({ metadata }) {
  if (!metadata) {
    return (
      <div className="card" style={{ marginBottom: '1.25rem' }}>
        <div className="card-header">
          <h3 className="card-title">
            <Info size={18} color="#2563eb" />
            <span>Raster Metadata Inspector</span>
          </h3>
        </div>
        <div style={{ padding: '1.5rem', textAlign: 'center', color: '#94a3b8', fontSize: '0.85rem' }}>
          No satellite raster loaded. Upload an image or select a benchmark scene above to inspect telemetry.
        </div>
      </div>
    );
  }

  return (
    <div className="card" style={{ marginBottom: '1.25rem' }}>
      <div className="card-header">
        <h3 className="card-title">
          <Info size={18} color="#2563eb" />
          <span>Raster Metadata Inspector</span>
        </h3>
        <Badge variant={metadata.is_multispectral ? 'ready' : 'pending'}>
          {metadata.is_multispectral ? '4-Band Multispectral' : `${metadata.channels}-Channel Optical`}
        </Badge>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
        <div className="meta-item">
          <span className="meta-label">File Name</span>
          <span className="meta-value" style={{ maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={metadata.filename}>
            {metadata.filename}
          </span>
        </div>
        <div className="meta-item">
          <span className="meta-label">File Size</span>
          <span className="meta-value">{metadata.file_size_formatted}</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Raster Dimensions</span>
          <span className="meta-value">{metadata.width} &times; {metadata.height} px</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Radiometric Channels</span>
          <span className="meta-value">
            {metadata.channels} Bands ({metadata.is_multispectral ? 'R, G, B, NIR' : 'RGB / Pan'})
          </span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Data Bit-Depth / Dtype</span>
          <span className="meta-value">{metadata.dtype}</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Spatial Reference (CRS)</span>
          <span className="meta-value">{metadata.crs || 'Local Pixel Grid (Non-Georeferenced)'}</span>
        </div>
        {metadata.resolution && (
          <div className="meta-item">
            <span className="meta-label">Pixel Ground Resolution</span>
            <span className="meta-value">{metadata.resolution[0].toFixed(2)} m / px</span>
          </div>
        )}
      </div>

      {/* Model Compatibility Matrix */}
      <div style={{ marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid #e2e8f0' }}>
        <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#475569', marginBottom: '0.5rem' }}>
          Architecture Compatibility:
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'JetBrains Mono, monospace' }}>
            <span>UNet-Mamba (CAVE-Flood)</span>
            <span style={{ color: metadata.compatibility?.unet_mamba ? '#059669' : '#dc2626', display: 'flex', alignItems: 'center', gap: '3px' }}>
              {metadata.compatibility?.unet_mamba ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />}
              {metadata.compatibility?.unet_mamba ? 'Compatible' : 'Incompatible'}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'JetBrains Mono, monospace' }}>
            <span>Standard U-Net</span>
            <span style={{ color: '#64748b' }}>Checkpoint Pending</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'JetBrains Mono, monospace' }}>
            <span>SegFormer-B2</span>
            <span style={{ color: '#64748b' }}>Checkpoint Pending</span>
          </div>
        </div>
      </div>
    </div>
  );
}
