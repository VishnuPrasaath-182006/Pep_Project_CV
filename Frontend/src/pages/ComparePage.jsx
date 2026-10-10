import React, { useState, useEffect } from 'react';
import { GitCompare, Sparkles, Play, AlertCircle, Clock, CheckCircle2 } from 'lucide-react';
import { api } from '../services/api';
import ImageUploader from '../components/upload/ImageUploader';
import LoadingSpinner from '../components/common/LoadingSpinner';

export default function ComparePage() {
  const [currentFile, setCurrentFile] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [isComparing, setIsComparing] = useState(false);
  const [compareData, setCompareData] = useState(null);
  const [error, setError] = useState(null);

  const handleImageLoaded = (meta, fileOrNull) => {
    setMetadata(meta);
    setCurrentFile(fileOrNull);
    setCompareData(null);
    setError(null);
  };

  const handleRunComparison = async () => {
    if (!metadata) {
      setError('Please select or upload a satellite image first.');
      return;
    }

    setIsComparing(true);
    setError(null);
    try {
      const res = await api.compare({
        file: currentFile,
        previewUrl: metadata.preview_url,
        modelIds: 'unet_mamba,unet,segformer',
      });
      setCompareData(res);
    } catch (err) {
      setError(err.response?.data?.detail || 'Comparison execution failed.');
    } finally {
      setIsComparing(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div>
        <h1 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#102a43' }}>
          Multi-Model Architecture Comparison Matrix
        </h1>
        <p style={{ fontSize: '0.85rem', color: '#64748b', marginTop: '2px' }}>
          Side-by-side evaluation across UNet-Mamba, Standard U-Net, and SegFormer
        </p>
      </div>

      {/* Top Input Bar */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ flex: 1, minWidth: '280px' }}>
            <ImageUploader
              onImageLoaded={handleImageLoaded}
              isLoading={isComparing}
              currentMetadata={metadata}
            />
          </div>

          <div style={{ alignSelf: 'flex-start', minWidth: '220px' }}>
            <button
              type="button"
              className="btn btn-primary"
              style={{ width: '100%', padding: '0.85rem' }}
              disabled={!metadata || isComparing}
              onClick={handleRunComparison}
            >
              <Play size={16} fill="#ffffff" />
              <span>{isComparing ? 'Executing Matrix...' : 'Run Comparative Matrix'}</span>
            </button>

            {metadata && (
              <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: '#64748b', fontFamily: 'JetBrains Mono, monospace' }}>
                Loaded: {metadata.filename} ({metadata.width}&times;{metadata.height} px)
              </div>
            )}
          </div>
        </div>

        {error && (
          <div style={{
            marginTop: '0.75rem',
            padding: '0.65rem 0.85rem',
            backgroundColor: '#fef2f2',
            border: '1px solid #fca5a5',
            borderRadius: '4px',
            color: '#991b1b',
            fontSize: '0.8rem',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}>
            <AlertCircle size={14} />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* In-Progress Spinner */}
      {isComparing && (
        <LoadingSpinner
          message="Executing Comparative Matrix..."
          submessage="Passing satellite raster through registered architectures..."
        />
      )}

      {/* Comparison Matrix Table & Visual Cards */}
      {!isComparing && compareData && (
        <div>
          <div className="grid-3" style={{ marginBottom: '1.5rem' }}>
            {compareData.models_evaluated.map((item) => {
              const res = item.result;
              return (
                <div key={item.model_id} className="card" style={{ display: 'flex', flexDirection: 'column' }}>
                  <div className="card-header">
                    <div>
                      <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#102a43' }}>
                        {item.model_name}
                      </h3>
                      <span style={{ fontSize: '0.7rem', color: '#64748b' }}>
                        {item.model_id}
                      </span>
                    </div>
                    {item.is_successful ? (
                      <span className="telemetry-chip chip-ready">
                        <CheckCircle2 size={11} /> Ready
                      </span>
                    ) : (
                      <span className="telemetry-chip chip-pending">
                        <Clock size={11} /> Checkpoint Pending
                      </span>
                    )}
                  </div>

                  {/* Visual Display */}
                  <div style={{ flex: 1, minHeight: '260px', backgroundColor: '#0f172a', borderRadius: '4px', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem' }}>
                    {item.is_successful && res ? (
                      <img
                        src={api.resolveAssetUrl(res.overlay_url)}
                        alt={`${item.model_name} overlay`}
                        style={{ maxWidth: '100%', maxHeight: '260px', objectFit: 'contain' }}
                      />
                    ) : (
                      <div style={{ textAlign: 'center', padding: '1.5rem', color: '#94a3b8' }}>
                        <Clock size={32} style={{ margin: '0 auto 0.5rem auto' }} />
                        <div style={{ fontWeight: 600, fontSize: '0.85rem', color: '#f1f5f9' }}>
                          Inference Offline
                        </div>
                        <div style={{ fontSize: '0.75rem', marginTop: '0.25rem', maxWidth: '200px' }}>
                          {item.error_message || 'Architecture preserved, trained weights pending.'}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Quantitative Metrics */}
                  <div style={{ fontSize: '0.8rem', display: 'flex', flexDirection: 'column', gap: '0.35rem', paddingTop: '0.5rem', borderTop: '1px solid #f1f5f9' }}>
                    <div className="meta-item">
                      <span className="meta-label">Latency</span>
                      <span className="meta-value">{res ? `${res.processing_time_ms} ms` : '—'}</span>
                    </div>
                    <div className="meta-item">
                      <span className="meta-label">Flood Coverage</span>
                      <span className="meta-value" style={{ color: res ? '#0284c7' : '#94a3b8' }}>
                        {res ? `${res.flood_coverage_percentage}%` : '—'}
                      </span>
                    </div>
                    <div className="meta-item">
                      <span className="meta-label">Inundated Pixels</span>
                      <span className="meta-value">
                        {res ? res.flood_pixels.toLocaleString() : '—'}
                      </span>
                    </div>
                    <div className="meta-item">
                      <span className="meta-label">Historical Benchmark</span>
                      <span className="meta-value">
                        {item.model_id === 'unet_mamba' ? '75.4% mIoU' :
                         item.model_id === 'unet' ? '73.3% mIoU' : '56.0% mIoU'}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Initial Guidance */}
      {!isComparing && !compareData && (
        <div className="card" style={{ padding: '3rem 2rem', textAlign: 'center' }}>
          <GitCompare size={36} color="#2563eb" style={{ margin: '0 auto 1rem auto' }} />
          <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#102a43', marginBottom: '0.5rem' }}>
            Multi-Model Evaluation Viewport
          </h2>
          <p style={{ fontSize: '0.9rem', color: '#64748b', maxWidth: '520px', margin: '0 auto' }}>
            Select an image above and click <strong>Run Comparative Matrix</strong> to evaluate
            the same satellite scene across all registered deep learning architectures simultaneously.
          </p>
        </div>
      )}
    </div>
  );
}
