import React, { useState } from 'react';
import { Eye, SplitSquareHorizontal, Layers, Image as ImageIcon, Sliders } from 'lucide-react';
import { api } from '../../services/api';
import ComparisonSlider from './ComparisonSlider';

export default function InferenceViewer({ result }) {
  const [activeTab, setActiveTab] = useState('split'); // 'split', 'overlay', 'mask', 'original'
  const [opacity, setOpacity] = useState(85);

  if (!result) return null;

  const originalUrl = api.resolveAssetUrl(result.original_preview_url);
  const maskUrl = api.resolveAssetUrl(result.mask_url);
  const overlayUrl = api.resolveAssetUrl(result.overlay_url);

  return (
    <div className="card" style={{ marginBottom: '1.25rem' }}>
      <div className="card-header" style={{ flexWrap: 'wrap', gap: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Layers size={18} color="#2563eb" />
          <h3 className="card-title" style={{ margin: 0 }}>
            Geospatial Segmentation Viewport
          </h3>
        </div>

        {/* View Mode Tabs */}
        <div style={{ display: 'flex', gap: '0.35rem', backgroundColor: '#f1f5f9', padding: '3px', borderRadius: '4px' }}>
          <button
            type="button"
            className={`btn btn-sm ${activeTab === 'split' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ border: 'none', padding: '3px 10px' }}
            onClick={() => setActiveTab('split')}
          >
            <SplitSquareHorizontal size={14} />
            <span>Split Slider</span>
          </button>
          <button
            type="button"
            className={`btn btn-sm ${activeTab === 'overlay' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ border: 'none', padding: '3px 10px' }}
            onClick={() => setActiveTab('overlay')}
          >
            <Eye size={14} />
            <span>Overlay</span>
          </button>
          <button
            type="button"
            className={`btn btn-sm ${activeTab === 'mask' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ border: 'none', padding: '3px 10px' }}
            onClick={() => setActiveTab('mask')}
          >
            <Sliders size={14} />
            <span>Binary Mask</span>
          </button>
          <button
            type="button"
            className={`btn btn-sm ${activeTab === 'original' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ border: 'none', padding: '3px 10px' }}
            onClick={() => setActiveTab('original')}
          >
            <ImageIcon size={14} />
            <span>Original</span>
          </button>
        </div>
      </div>

      {/* Opacity Control for Overlay View */}
      {activeTab === 'overlay' && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '1rem',
          padding: '0.5rem 1rem',
          backgroundColor: '#f8fafc',
          borderBottom: '1px solid #e2e8f0',
          marginBottom: '0.75rem',
          borderRadius: '4px'
        }}>
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#475569' }}>
            Flood Mask Opacity:
          </span>
          <input
            type="range"
            min="10"
            max="100"
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            style={{ flex: 1, accentColor: '#2563eb' }}
          />
          <span style={{ fontSize: '0.8rem', fontFamily: 'JetBrains Mono, monospace', minWidth: '40px', fontWeight: 600 }}>
            {opacity}%
          </span>
        </div>
      )}

      {/* Main Visual Display Canvas */}
      <div className="viewer-container">
        {activeTab === 'split' && (
          <ComparisonSlider originalSrc={originalUrl} overlaySrc={overlayUrl} />
        )}

        {activeTab === 'overlay' && (
          <div style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', justifyContent: 'center' }}>
            <img
              src={originalUrl}
              alt="Original"
              style={{ maxWidth: '100%', maxHeight: '520px', objectFit: 'contain', display: 'block' }}
            />
            <div
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                opacity: opacity / 100.0,
                display: 'flex',
                justifyContent: 'center',
                pointerEvents: 'none'
              }}
            >
              <img
                src={overlayUrl}
                alt="Overlay"
                style={{ maxWidth: '100%', maxHeight: '520px', objectFit: 'contain' }}
              />
            </div>
          </div>
        )}

        {activeTab === 'mask' && (
          <img
            src={maskUrl}
            alt="Predicted binary flood segmentation mask"
            className="viewer-image"
          />
        )}

        {activeTab === 'original' && (
          <img
            src={originalUrl}
            alt="Original optical satellite raster"
            className="viewer-image"
          />
        )}
      </div>

      {/* Color-Coded Semantic Class Taxonomy Legend */}
      <div style={{
        marginTop: '0.75rem',
        padding: '0.65rem 1rem',
        backgroundColor: '#f8fafc',
        borderRadius: '4px',
        border: '1px solid #e2e8f0',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '0.5rem',
        fontSize: '0.78rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '12px', height: '12px', backgroundColor: '#0ea5e9', borderRadius: '2px', display: 'inline-block' }} />
            <span style={{ fontWeight: 600, color: '#0369a1' }}>Class 1: Flood Inundation</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '12px', height: '12px', backgroundColor: '#1e293b', borderRadius: '2px', display: 'inline-block' }} />
            <span style={{ color: '#64748b' }}>Class 0: Non-Flood / Dry Ground</span>
          </div>
        </div>

        <div style={{ fontFamily: 'JetBrains Mono, monospace', color: '#64748b' }}>
          Resolution: {result.image_width} &times; {result.image_height} px &bull; Decision Threshold: 0.50
        </div>
      </div>
    </div>
  );
}
