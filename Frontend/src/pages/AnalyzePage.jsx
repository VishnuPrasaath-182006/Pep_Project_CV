import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play, Sparkles, AlertCircle, RefreshCw } from 'lucide-react';
import { api } from '../services/api';
import ImageUploader from '../components/upload/ImageUploader';
import MetadataInspector from '../components/upload/MetadataInspector';
import ModelSelector from '../components/upload/ModelSelector';
import InferenceViewer from '../components/results/InferenceViewer';
import MetricsPanel from '../components/results/MetricsPanel';
import ExportToolbar from '../components/results/ExportToolbar';
import LoadingSpinner from '../components/common/LoadingSpinner';

export default function AnalyzePage() {
  const navigate = useNavigate();
  const [currentFile, setCurrentFile] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [selectedModel, setSelectedModel] = useState('unet_mamba');
  const [isInferring, setIsInferring] = useState(false);
  const [inferenceStep, setInferenceStep] = useState('');
  const [inferenceResult, setInferenceResult] = useState(null);
  const [error, setError] = useState(null);

  const handleImageLoaded = (meta, fileOrNull) => {
    setMetadata(meta);
    setCurrentFile(fileOrNull);
    setInferenceResult(null);
    setError(null);
  };

  const handleRunInference = async () => {
    if (!metadata) {
      setError('Please upload or load a satellite image first.');
      return;
    }

    setIsInferring(true);
    setError(null);
    setInferenceStep('Normalizing multi-band raster and preparing 512x512 tiles...');

    try {
      setTimeout(() => {
        setInferenceStep('Executing 2D Selective State Space forward pass...');
      }, 1200);

      const result = await api.predict({
        file: currentFile,
        modelId: selectedModel,
        previewUrl: metadata.preview_url,
      });

      setInferenceStep('Synthesizing flood mask and georeferenced GeoTIFF...');
      setInferenceResult(result);
      navigate(`/results/${result.result_id}`);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        'Inference failed. Check image format or model compatibility.'
      );
    } finally {
      setIsInferring(false);
      setInferenceStep('');
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Workspace Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <h1 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#102a43' }}>
            Satellite Image Analysis Workspace
          </h1>
          <p style={{ fontSize: '0.85rem', color: '#64748b', marginTop: '2px' }}>
            High-precision semantic segmentation for flood inundation extent identification
          </p>
        </div>

        {inferenceResult && (
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => {
              setInferenceResult(null);
            }}
          >
            <RefreshCw size={14} />
            <span>Reset Viewport</span>
          </button>
        )}
      </div>

      {/* Main Dual-Column Responsive Grid */}
      <div className="grid-sidebar">
        {/* Left Column: Ingestion, Inspection & Configuration */}
        <div>
          <ImageUploader
            onImageLoaded={handleImageLoaded}
            isLoading={isInferring}
            currentMetadata={metadata}
          />

          <MetadataInspector metadata={metadata} />

          <ModelSelector
            selectedModel={selectedModel}
            onSelectModel={(id) => setSelectedModel(id)}
            disabled={isInferring}
          />

          {/* Primary Action Button */}
          <button
            type="button"
            className="btn btn-primary"
            style={{ width: '100%', padding: '0.85rem', fontSize: '1rem', marginTop: '0.5rem' }}
            disabled={!metadata || isInferring}
            onClick={handleRunInference}
          >
            {isInferring ? (
              <span>Processing Inference...</span>
            ) : (
              <>
                <Play size={18} fill="#ffffff" />
                <span>Run Flood Detection</span>
              </>
            )}
          </button>

          {error && (
            <div style={{
              marginTop: '1rem',
              padding: '0.75rem 1rem',
              backgroundColor: '#fef2f2',
              border: '1px solid #fca5a5',
              borderRadius: '4px',
              color: '#991b1b',
              fontSize: '0.825rem',
              display: 'flex',
              alignItems: 'flex-start',
              gap: '8px'
            }}>
              <AlertCircle size={16} style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <div style={{ fontWeight: 600 }}>Inference Request Error</div>
                <div>{error}</div>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Dynamic Viewport & Results */}
        <div>
          {isInferring && (
            <LoadingSpinner
              message="Executing PyTorch Semantic Segmentation..."
              submessage={inferenceStep}
            />
          )}

          {!isInferring && inferenceResult && (
            <div>
              <InferenceViewer result={inferenceResult} />
              <MetricsPanel result={inferenceResult} />
              <ExportToolbar result={inferenceResult} />
            </div>
          )}

          {!isInferring && !inferenceResult && metadata && (
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">
                  <span>Loaded Scene Preview</span>
                </h3>
                <span style={{ fontSize: '0.75rem', color: '#64748b', fontFamily: 'JetBrains Mono, monospace' }}>
                  {metadata.width} &times; {metadata.height} px
                </span>
              </div>
              <div className="viewer-container">
                <img
                  src={api.resolveAssetUrl(metadata.preview_url)}
                  alt="Loaded satellite preview"
                  className="viewer-image"
                />
              </div>
              <div style={{ marginTop: '1rem', textAlign: 'center' }}>
                <p style={{ fontSize: '0.85rem', color: '#475569', marginBottom: '0.75rem' }}>
                  Satellite scene is verified and loaded. Click <strong>Run Flood Detection</strong> to execute inference.
                </p>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={handleRunInference}
                >
                  <Play size={16} fill="#ffffff" />
                  <span>Execute Analysis</span>
                </button>
              </div>
            </div>
          )}

          {!isInferring && !inferenceResult && !metadata && (
            <div className="card" style={{ padding: '3rem 2rem', textAlign: 'center' }}>
              <div style={{
                width: '56px',
                height: '56px',
                borderRadius: '50%',
                backgroundColor: '#eff6ff',
                color: '#2563eb',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 1.25rem auto'
              }}>
                <Sparkles size={28} />
              </div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#102a43', marginBottom: '0.5rem' }}>
                Analysis Viewport Awaiting Satellite Scene
              </h2>
              <p style={{ fontSize: '0.9rem', color: '#64748b', maxWidth: '480px', margin: '0 auto 1.5rem auto', lineHeight: 1.5 }}>
                Upload a 4-band PlanetScope GeoTIFF raster, optical RGB image, or choose one of our
                authentic benchmark test scenes on the left to begin flood extent identification.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
