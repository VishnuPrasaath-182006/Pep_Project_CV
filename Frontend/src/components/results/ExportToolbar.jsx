import React from 'react';
import { Download, FileDown, Layers, FileJson } from 'lucide-react';
import { api } from '../../services/api';

export default function ExportToolbar({ result }) {
  if (!result) return null;

  const maskDownloadUrl = api.getDownloadUrl(result.result_id, 'mask');
  const overlayDownloadUrl = api.getDownloadUrl(result.result_id, 'overlay');
  const geotiffDownloadUrl = api.getDownloadUrl(result.result_id, 'geotiff');
  const reportDownloadUrl = api.getDownloadUrl(result.result_id, 'report');

  return (
    <div className="card" style={{ marginBottom: '1.25rem' }}>
      <div className="card-header">
        <h3 className="card-title">
          <Download size={18} color="#2563eb" />
          <span>Scientific Results Export Pipeline</span>
        </h3>
        <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
          Format Verification & CRS Preservation
        </span>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem' }}>
        <a
          href={maskDownloadUrl}
          download
          className="btn btn-secondary btn-sm"
          style={{ textDecoration: 'none' }}
        >
          <FileDown size={14} color="#2563eb" />
          <span>Download Binary Mask (PNG)</span>
        </a>

        <a
          href={overlayDownloadUrl}
          download
          className="btn btn-secondary btn-sm"
          style={{ textDecoration: 'none' }}
        >
          <Layers size={14} color="#0ea5e9" />
          <span>Download Flood Overlay (PNG)</span>
        </a>

        <a
          href={geotiffDownloadUrl}
          download
          className="btn btn-secondary btn-sm"
          style={{ textDecoration: 'none' }}
        >
          <FileDown size={14} color="#059669" />
          <span>Download GeoTIFF Mask (.tif)</span>
        </a>

        <a
          href={reportDownloadUrl}
          download
          className="btn btn-secondary btn-sm"
          style={{ textDecoration: 'none' }}
        >
          <FileJson size={14} color="#f59e0b" />
          <span>Download JSON Audit Report</span>
        </a>
      </div>
    </div>
  );
}
