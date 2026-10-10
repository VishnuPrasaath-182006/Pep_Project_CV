import React, { useState, useEffect } from 'react';
import { BookOpen, ShieldCheck, Cpu, HardDrive, CheckCircle2, Clock } from 'lucide-react';
import { api } from '../services/api';
import Badge from '../components/common/Badge';

export default function ModelsPage() {
  const [models, setModels] = useState([]);
  const [checkpoints, setCheckpoints] = useState([]);

  useEffect(() => {
    api.getModels()
      .then(res => setModels(res.models))
      .catch(() => {});
    api.getCheckpoints()
      .then(res => setCheckpoints(res.checkpoints))
      .catch(() => {});
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <h1 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#102a43' }}>
          Deep Learning Architectures &amp; Scientific Specifications
        </h1>
        <p style={{ fontSize: '0.85rem', color: '#64748b', marginTop: '2px' }}>
          Detailed technical formulations, input constraints, and cryptographic checkpoint integrity ledger
        </p>
      </div>

      {/* Model Specifications Section */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        {/* UNet-Mamba */}
        <div className="card">
          <div className="card-header">
            <div>
              <h2 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#102a43' }}>
                1. UNet-Mamba (CAVE-Flood)
              </h2>
              <span style={{ fontSize: '0.8rem', color: '#2563eb', fontWeight: 500 }}>
                2D Selective State Space (SS2D) + ResT Efficient Spatial Encoder
              </span>
            </div>
            <Badge variant="ready">
              <CheckCircle2 size={12} />
              <span>Active Trained Checkpoint (171 MB)</span>
            </Badge>
          </div>

          <div style={{ fontSize: '0.875rem', color: '#334155', lineHeight: 1.6, marginBottom: '1rem' }}>
            <p style={{ marginBottom: '0.75rem' }}>
              <strong>Architecture Overview:</strong> UNet-Mamba integrates an efficient ResT multi-scale
              backbone encoder with a state-space segmentation decoder powered by 2D Selective Scan (SS2D).
              Unlike standard Vision Transformers which incur quadratic &Omicron;(N&sup2;) complexity with respect
              to raster token length, Mamba achieves linear &Omicron;(N) complexity, enabling unconstrained
              receptive fields and continuous hydrological boundary tracking.
            </p>
            <p style={{ marginBottom: '0.75rem' }}>
              <strong>Input Normalization:</strong> Designed specifically for PlanetScope 4-band multispectral rasters
              [Red, Green, Blue, Near-Infrared]. Applies channel-wise z-score normalization:
            </p>
            <div style={{
              backgroundColor: '#f8fafc',
              padding: '0.75rem 1rem',
              borderRadius: '4px',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '0.8rem',
              border: '1px solid #e2e8f0',
              marginBottom: '0.75rem'
            }}>
              Mean: [2334.81, 2378.43, 2185.12, 3608.97]<br />
              Std:  [1245.52, 1184.21, 1148.65, 1420.33]
            </div>
            <p>
              <strong>Supervision &amp; Losses:</strong> Trained using a compound loss formulation combining
              Soft Cross-Entropy with label smoothing (0.05), Dice Loss, and auxiliary supervision via Local Supervision Modules (LSM).
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid #f1f5f9' }}>
            <div className="meta-item">
              <span className="meta-label">Test Mean IoU</span>
              <span className="meta-value" style={{ color: '#059669' }}>75.42%</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Flood Dice / F1</span>
              <span className="meta-value" style={{ color: '#059669' }}>87.37%</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Flood Precision</span>
              <span className="meta-value">91.69%</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Pixel Accuracy</span>
              <span className="meta-value">86.11%</span>
            </div>
          </div>
        </div>

        {/* Standard U-Net */}
        <div className="card">
          <div className="card-header">
            <div>
              <h2 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#102a43' }}>
                2. Standard U-Net (Baseline)
              </h2>
              <span style={{ fontSize: '0.8rem', color: '#0f766e', fontWeight: 500 }}>
                Convolutional Encoder-Decoder with Symmetric Skip Connections
              </span>
            </div>
            <Badge variant="pending">
              <Clock size={12} />
              <span>Preserved Architecture &bull; Weights Pending</span>
            </Badge>
          </div>

          <div style={{ fontSize: '0.875rem', color: '#334155', lineHeight: 1.6, marginBottom: '1rem' }}>
            <p style={{ marginBottom: '0.75rem' }}>
              <strong>Architecture Overview:</strong> Classic symmetric 4-stage convolutional neural network.
              The contracting encoder path extracts multiscale contextual representations using repeated 3&times;3
              convolutions, batch normalization, and ReLU activations followed by max pooling. The expansive
              decoder path upsamples spatial dimensions and concatenates high-resolution encoder features via lateral skip connections.
            </p>
            <p>
              <strong>Dataset Role:</strong> Established benchmark baseline for evaluating convolutional spatial feature
              preservation versus attention- and state-space-based models on PlanetScope imagery.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid #f1f5f9' }}>
            <div className="meta-item">
              <span className="meta-label">Historical Mean IoU</span>
              <span className="meta-value">73.26%</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Historical F1 Score</span>
              <span className="meta-value">84.17%</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Historical Precision</span>
              <span className="meta-value">85.39%</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Historical Recall</span>
              <span className="meta-value">83.45%</span>
            </div>
          </div>
        </div>

        {/* SegFormer-B2 */}
        <div className="card">
          <div className="card-header">
            <div>
              <h2 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#102a43' }}>
                3. SegFormer-B2
              </h2>
              <span style={{ fontSize: '0.8rem', color: '#0ea5e9', fontWeight: 500 }}>
                Hierarchical Vision Transformer (MiT-B2) + All-MLP Decoder
              </span>
            </div>
            <Badge variant="pending">
              <Clock size={12} />
              <span>Preserved Architecture &bull; Weights Pending</span>
            </Badge>
          </div>

          <div style={{ fontSize: '0.875rem', color: '#334155', lineHeight: 1.6, marginBottom: '1rem' }}>
            <p style={{ marginBottom: '0.75rem' }}>
              <strong>Architecture Overview:</strong> Vision transformer designed for semantic segmentation without
              fixed positional embeddings, enabling arbitrary input resolutions without spatial degradation.
              The Mix Transformer (MiT-B2) hierarchical encoder generates multiscale features that are merged by a
              lightweight All-MLP decoder.
            </p>
            <p>
              <strong>Evaluation Protocol:</strong> Audited across a rigorous 19-fold Leave-One-Event-Out (LOEO)
              cross-validation benchmark spanning 383+ million held-out satellite pixels.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid #f1f5f9' }}>
            <div className="meta-item">
              <span className="meta-label">19-Fold Macro mIoU</span>
              <span className="meta-value">55.98%</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Flood IoU</span>
              <span className="meta-value">55.35%</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Dice / F1</span>
              <span className="meta-value">70.55%</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Pixel Accuracy</span>
              <span className="meta-value">73.33%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Cryptographic Checkpoint Ledger Table */}
      <div className="card">
        <div className="card-header">
          <h2 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#102a43' }}>
            Cryptographic Checkpoint Integrity Ledger
          </h2>
          <span style={{ fontSize: '0.75rem', fontFamily: 'JetBrains Mono, monospace', color: '#64748b' }}>
            SHA-256 Verified
          </span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'left' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', color: '#475569' }}>
                <th style={{ padding: '0.65rem 0.75rem' }}>Model Architecture</th>
                <th style={{ padding: '0.65rem 0.75rem' }}>Artifact Filename</th>
                <th style={{ padding: '0.65rem 0.75rem' }}>File Size</th>
                <th style={{ padding: '0.65rem 0.75rem' }}>SHA-256 Checksum</th>
                <th style={{ padding: '0.65rem 0.75rem' }}>Deployment Status</th>
              </tr>
            </thead>
            <tbody>
              {checkpoints.length > 0 ? (
                checkpoints.map(item => {
                  const isReady = item.deployment_readiness.includes('Ready');
                  return (
                    <tr key={item.model_id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '0.65rem 0.75rem', fontWeight: 600 }}>{item.model_name}</td>
                      <td style={{ padding: '0.65rem 0.75rem', fontFamily: 'JetBrains Mono, monospace' }}>
                        {item.artifact_filename}
                      </td>
                      <td style={{ padding: '0.65rem 0.75rem', fontFamily: 'JetBrains Mono, monospace', color: isReady ? '#102a43' : '#94a3b8' }}>
                        {item.file_size_formatted}
                      </td>
                      <td style={{ padding: '0.65rem 0.75rem', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.72rem', color: item.sha256_checksum ? '#102a43' : '#94a3b8' }}>
                        {item.sha256_checksum ? `${item.sha256_checksum.substring(0, 32)}...` : '—'}
                      </td>
                      <td style={{ padding: '0.65rem 0.75rem' }}>
                        <Badge variant={isReady ? 'ready' : 'pending'}>
                          {item.deployment_readiness}
                        </Badge>
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td colSpan={5} style={{ padding: '1rem', textAlign: 'center', color: '#64748b' }}>
                    Loading cryptographic ledger from FastAPI backend...
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
