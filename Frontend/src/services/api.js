import axios from 'axios';

// Base URL resolution: uses Vite env if supplied, else relies on Vite proxy '/api'
const BASE_URL = import.meta.env.VITE_API_BASE_URL 
  ? `${import.meta.env.VITE_API_BASE_URL}/api`
  : '/api';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 180000, // 3 minutes timeout for CPU/GPU model inference
});

export const api = {
  // System Health
  async getHealth() {
    const res = await apiClient.get('/health');
    return res.data;
  },

  // Models Info & Registry
  async getModels() {
    const res = await apiClient.get('/models');
    return res.data;
  },

  // Cryptographic Checkpoints Ledger
  async getCheckpoints() {
    const res = await apiClient.get('/checkpoints');
    return res.data;
  },

  // Image Upload & Inspection
  async inspectImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await apiClient.post('/image/inspect', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  // Sample Scenes
  async getSampleChips() {
    const res = await apiClient.get('/image/samples');
    return res.data;
  },

  async loadSampleChip(sampleId) {
    const res = await apiClient.post(`/image/load-sample?sample_id=${encodeURIComponent(sampleId)}`);
    return res.data;
  },

  // Single Model Inference
  async predict({ file, modelId = 'unet_mamba', previewUrl = null }) {
    const formData = new FormData();
    if (file) {
      formData.append('file', file);
    }
    formData.append('model_id', modelId);
    if (previewUrl) {
      formData.append('preview_url', previewUrl);
    }

    const res = await apiClient.post('/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  // Multi-Model Comparison
  async compare({ file, previewUrl = null, modelIds = 'unet_mamba,unet,segformer' }) {
    const formData = new FormData();
    if (file) {
      formData.append('file', file);
    }
    if (previewUrl) {
      formData.append('preview_url', previewUrl);
    }
    formData.append('model_ids', modelIds);

    const res = await apiClient.post('/compare', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  // Retrieve Result
  async getResult(resultId) {
    const res = await apiClient.get(`/results/${resultId}`);
    return res.data;
  },

  // Asset Download URL
  getDownloadUrl(resultId, assetType) {
    return `${BASE_URL}/download/${resultId}/${assetType}`;
  },

  // Format asset URL (for preview, mask, overlay images)
  resolveAssetUrl(url) {
    if (!url) return '';
    if (url.startsWith('http')) return url;
    if (import.meta.env.VITE_API_BASE_URL) {
      return `${import.meta.env.VITE_API_BASE_URL}${url}`;
    }
    return url;
  }
};
