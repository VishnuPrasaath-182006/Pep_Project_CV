import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, AlertCircle } from 'lucide-react';
import { api } from '../services/api';
import InferenceViewer from '../components/results/InferenceViewer';
import MetricsPanel from '../components/results/MetricsPanel';
import ExportToolbar from '../components/results/ExportToolbar';
import LoadingSpinner from '../components/common/LoadingSpinner';

export default function ResultsPage() {
  const { id } = useParams();
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!id) return;
    setIsLoading(true);
    api.getResult(id)
      .then(data => setResult(data))
      .catch(err => {
        setError(err.response?.data?.detail || 'Inference result not found.');
      })
      .finally(() => setIsLoading(false));
  }, [id]);

  if (isLoading) {
    return <LoadingSpinner message="Retrieving cached inference results..." />;
  }

  if (error || !result) {
    return (
      <div className="card" style={{ padding: '3rem 2rem', textAlign: 'center' }}>
        <AlertCircle size={36} color="#ef4444" style={{ margin: '0 auto 1rem auto' }} />
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#102a43', marginBottom: '0.5rem' }}>
          Result Not Found
        </h2>
        <p style={{ fontSize: '0.9rem', color: '#64748b', marginBottom: '1.5rem' }}>
          {error || 'Unable to locate the specified prediction identifier.'}
        </p>
        <Link to="/analyze" className="btn btn-primary">
          <ArrowLeft size={16} />
          <span>Return to Analysis</span>
        </Link>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <Link to="/analyze" className="btn btn-secondary btn-sm">
            <ArrowLeft size={14} />
            <span>Back to Analysis</span>
          </Link>
          <h1 style={{ fontSize: '1.3rem', fontWeight: 700, color: '#102a43' }}>
            Inference Record: <span style={{ fontFamily: 'JetBrains Mono, monospace' }}>{result.result_id}</span>
          </h1>
        </div>
      </div>

      <InferenceViewer result={result} />
      <MetricsPanel result={result} />
      <ExportToolbar result={result} />
    </div>
  );
}
