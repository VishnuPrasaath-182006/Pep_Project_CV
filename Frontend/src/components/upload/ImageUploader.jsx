import React, { useState, useEffect, useRef } from 'react';
import { UploadCloud, FileImage, Sparkles, AlertCircle, CheckCircle2 } from 'lucide-react';
import { api } from '../../services/api';

export default function ImageUploader({ onImageLoaded, isLoading, currentMetadata }) {
  const [isDragging, setIsDragging] = useState(false);
  const [sampleChips, setSampleChips] = useState([]);
  const [loadingSample, setLoadingSample] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    api.getSampleChips()
      .then(samples => setSampleChips(samples))
      .catch(() => {});
  }, []);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  };

  const handleFileSelected = async (file) => {
    setError(null);
    try {
      const meta = await api.inspectImage(file);
      onImageLoaded(meta, file);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to inspect satellite raster. Check file format.');
    }
  };

  const handleSelectSample = async (sampleId) => {
    setError(null);
    setLoadingSample(true);
    try {
      const meta = await api.loadSampleChip(sampleId);
      onImageLoaded(meta, null);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load sample satellite chip.');
    } finally {
      setLoadingSample(false);
    }
  };

  return (
    <div className="card" style={{ marginBottom: '1.25rem' }}>
      <div className="card-header">
        <h3 className="card-title">
          <UploadCloud size={18} color="#2563eb" />
          <span>Satellite Raster Ingestion</span>
        </h3>
        <span style={{ fontSize: '0.75rem', fontFamily: 'JetBrains Mono, monospace', color: '#64748b' }}>
          GeoTIFF / PNG / JPG / NPZ
        </span>
      </div>

      <div
        className={`dropzone ${isDragging ? 'active' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".tif,.tiff,.geotiff,.png,.jpg,.jpeg,.npz"
          style={{ display: 'none' }}
          onChange={handleFileChange}
          disabled={isLoading || loadingSample}
        />
        
        <FileImage size={38} color="#2563eb" style={{ margin: '0 auto 0.75rem auto' }} />
        <div style={{ fontWeight: 600, fontSize: '0.95rem', color: '#102a43', marginBottom: '0.25rem' }}>
          Drag & Drop Satellite Image Here
        </div>
        <div style={{ fontSize: '0.8rem', color: '#64748b', marginBottom: '1rem' }}>
          Supports 4-band Multispectral PlanetScope rasters or Optical RGB imagery (up to 100 MB)
        </div>
        
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          disabled={isLoading || loadingSample}
          onClick={(e) => {
            e.stopPropagation();
            fileInputRef.current?.click();
          }}
        >
          Select File from Device
        </button>
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

      {/* Quick Load Research Samples */}
      {sampleChips.length > 0 && (
        <div style={{ marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid #e2e8f0' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#475569', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Sparkles size={13} color="#2563eb" />
            <span>Load Authentic Benchmark Scene:</span>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {sampleChips.map((chip, idx) => (
              <button
                key={chip.id}
                type="button"
                className="btn btn-secondary btn-sm"
                style={{ fontSize: '0.75rem', fontFamily: 'JetBrains Mono, monospace' }}
                disabled={isLoading || loadingSample}
                onClick={() => handleSelectSample(chip.id)}
              >
                <span>Scene {idx + 1} (Bangladesh 4-Band)</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
