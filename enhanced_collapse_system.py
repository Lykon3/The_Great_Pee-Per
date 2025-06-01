#!/usr/bin/env python3
"""
Enhanced Universal Collapse Control System
==========================================
Production-ready implementation with advanced mathematical models,
real-time streaming, and multi-domain intervention capabilities.

Key Features:
- Multi-phase spectral flow analysis
- Tensor mechanics with eigenmode tracking
- Quantum friction modeling
- Real-time intervention protocols
- Universal signature extraction
- Cross-domain pattern recognition
"""

import numpy as np
import pandas as pd
from scipy import signal, optimize, interpolate, linalg
from scipy.special import gamma, digamma
from scipy.stats import entropy as scipy_entropy
import pywt
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import time
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')

@dataclass
class UniversalSignature:
    """Universal collapse signature with enhanced components"""
    timestamp: datetime
    domain: str
    
    # Spectral components
    phase_count: int
    critical_times: List[float]
    steepness_ratios: List[float]
    saturation_pattern: List[float]
    
    # Topological invariants
    dimensional_reduction_factor: float
    eigenmode_collapse_rate: float
    topology_shift_indicator: float
    
    # Entropy dynamics
    entropy_production_rate: float
    entropy_reversal_point: Optional[float]
    
    # Universal exponents
    alpha_exponent: float  # ρc ~ (τc - τ)^(-α)
    beta_exponent: float   # λmax ~ (τc - τ)^β
    gamma_exponent: float  # D ~ (τc - τ)^γ
    
    # Quantum friction effects
    charge_tunneling_rate: float
    fractal_dimension: float
    quantum_friction_coeff: float
    
    # Intervention metrics
    intervention_window: Tuple[float, float]
    intervention_priority: float
    optimal_intervention_type: str
    
    # Confidence metrics
    model_confidence: float
    prediction_uncertainty: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedMathematicalModels:
    """Enhanced mathematical models for collapse detection"""
    
    @staticmethod
    def multi_phase_sigmoid(t: np.ndarray, phases: List[Dict]) -> np.ndarray:
        """Multi-phase sigmoid with spectral flow saturation"""
        result = np.zeros_like(t)
        
        for phase in phases:
            w, k, tau_c = phase['weight'], phase['steepness'], phase['critical_time']
            sigmoid = w / (1 + np.exp(-k * (t - tau_c)))
            
            # Add spectral flow correction
            if 'gradient_factor' in phase:
                gradient_correction = 1 + phase['gradient_factor'] * np.gradient(sigmoid)
                sigmoid *= gradient_correction
            
            result += sigmoid
        
        return result
    
    @staticmethod
    def tensor_eigenmode_evolution(tensor_field: np.ndarray, dt: float, 
                                  gamma_decay: np.ndarray, coupling_matrix: np.ndarray,
                                  ricci_coupling: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evolve eigenmode dynamics with Ricci flow correction"""
        eigenvals, eigenvecs = linalg.eigh(tensor_field)
        
        # Mode evolution equation: dλₙ/dτ = -γₙλₙ + βₙΣₘ Cₙₘλₘ - αₙR
        mode_coupling = coupling_matrix @ eigenvals
        ricci_effect = ricci_coupling * np.trace(tensor_field)
        
        d_eigenvals = -gamma_decay * eigenvals + mode_coupling - ricci_effect
        new_eigenvals = eigenvals + dt * d_eigenvals
        
        # Reconstruct tensor
        new_tensor = eigenvecs @ np.diag(new_eigenvals) @ eigenvecs.T
        
        return new_tensor, new_eigenvals
    
    @staticmethod
    def quantum_friction_rate(energy: np.ndarray, fractal_dim: float, 
                             temperature: float = 1.0) -> np.ndarray:
        """Compute quantum friction with fractal boundary corrections"""
        hbar_over_2pi = 1.0  # Normalized units
        
        # Fractal-corrected tunneling
        kappa = 1.0  # Barrier parameter
        d_eff = energy ** (1.0 / fractal_dim)
        
        tunneling_factor = np.exp(-2 * kappa * d_eff)
        
        # Thermal factor
        thermal_factor = np.tanh(energy / (2 * temperature))
        
        return hbar_over_2pi * tunneling_factor * thermal_factor
    
    @staticmethod
    def fractal_dimension_estimation(data: np.ndarray, scales: Optional[np.ndarray] = None) -> float:
        """Estimate fractal dimension using box-counting method"""
        if scales is None:
            scales = np.logspace(0.5, 2, 20)
        
        # Normalize data to [0, 1]
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        counts = []
        for scale in scales:
            # Grid-based box counting
            grid_size = int(len(data_norm) / scale)
            if grid_size < 2:
                continue
                
            grid = np.linspace(0, 1, grid_size)
            boxes = np.histogram(data_norm, bins=grid)[0]
            count = np.sum(boxes > 0)
            counts.append(count)
        
        if len(counts) < 3:
            return 1.0  # Default dimension
        
        # Fit log-log relationship
        log_scales = np.log(scales[:len(counts)])
        log_counts = np.log(counts)
        
        slope, _ = np.polyfit(log_scales, log_counts, 1)
        fractal_dim = -slope
        
        return max(0.1, min(3.0, fractal_dim))  # Bound dimension

class EnhancedFeatureExtractor:
    """Advanced feature extraction with topological and quantum components"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def extract_spectral_features(self, data: np.ndarray, timestamps: np.ndarray) -> Dict:
        """Extract spectral acceleration and saturation features"""
        features = {}
        
        # Compute second derivative (acceleration)
        if len(data) >= 3:
            acceleration = np.abs(np.gradient(np.gradient(data)))
            features['max_acceleration'] = np.max(acceleration)
            features['acceleration_variance'] = np.var(acceleration)
            
            # Find acceleration peaks
            peaks, _ = signal.find_peaks(acceleration, height=np.max(acceleration) * 0.3)
            features['acceleration_peaks'] = len(peaks)
            
            # Saturation metric
            if len(peaks) > 0:
                saturation = np.cumsum(acceleration) / np.max(np.cumsum(acceleration))
                features['saturation_rate'] = np.max(np.gradient(saturation))
            else:
                features['saturation_rate'] = 0.0
        
        return features
    
    def extract_topological_features(self, data: np.ndarray) -> Dict:
        """Extract topological invariants and connectivity metrics"""
        features = {}
        
        # Embedding dimension estimation using False Nearest Neighbors
        embedding_dim = self._estimate_embedding_dimension(data)
        features['embedding_dimension'] = embedding_dim
        
        # Persistent homology approximation
        # (Simplified implementation - full version would use gudhi/ripser)
        if len(data) > 10:
            # Distance matrix for simplicial complex
            n_points = min(100, len(data))  # Subsample for efficiency
            indices = np.linspace(0, len(data)-1, n_points, dtype=int)
            subset = data[indices]
            
            dist_matrix = np.abs(subset[:, None] - subset[None, :])
            threshold = np.percentile(dist_matrix, 20)
            
            # Connected components (Betti-0)
            adjacency = dist_matrix < threshold
            features['betti_0'] = self._count_connected_components(adjacency)
            
            # Approximate Betti-1 (cycles)
            features['betti_1'] = max(0, np.sum(adjacency) - len(subset) - features['betti_0'] + 1)
        
        return features
    
    def extract_quantum_features(self, data: np.ndarray) -> Dict:
        """Extract quantum friction and tunneling features"""
        features = {}
        
        # Fractal dimension
        fractal_dim = AdvancedMathematicalModels.fractal_dimension_estimation(data)
        features['fractal_dimension'] = fractal_dim
        
        # Energy landscape analysis
        potential = -np.cumsum(data - np.mean(data))
        energy_levels = np.sort(potential)
        
        # Quantum tunneling rates
        if len(energy_levels) > 1:
            energy_gaps = np.diff(energy_levels)
            tunneling_rates = AdvancedMathematicalModels.quantum_friction_rate(
                energy_gaps, fractal_dim
            )
            features['mean_tunneling_rate'] = np.mean(tunneling_rates)
            features['max_tunneling_rate'] = np.max(tunneling_rates)
        
        # Quantum coherence measure
        fft_data = np.fft.fft(data)
        coherence = np.abs(fft_data) ** 2
        coherence /= np.sum(coherence)
        features['quantum_coherence'] = -np.sum(coherence * np.log(coherence + 1e-10))
        
        return features
    
    def extract_multiscale_entropy(self, data: np.ndarray, scales: List[int] = None) -> Dict:
        """Multi-scale entropy analysis"""
        if scales is None:
            scales = [1, 2, 4, 8, 16]
        
        features = {}
        entropies = []
        
        for scale in scales:
            if len(data) >= scale * 10:  # Need sufficient data points
                # Coarse-grain the series
                coarse_grained = self._coarse_grain(data, scale)
                
                # Sample entropy
                sample_ent = self._sample_entropy(coarse_grained)
                entropies.append(sample_ent)
        
        if entropies:
            features['multiscale_entropy_mean'] = np.mean(entropies)
            features['multiscale_entropy_slope'] = self._fit_entropy_slope(scales[:len(entropies)], entropies)
            features['entropy_complexity'] = np.sum(np.diff(entropies) ** 2)
        
        return features
    
    def _estimate_embedding_dimension(self, data: np.ndarray, max_dim: int = 10) -> int:
        """Estimate embedding dimension using False Nearest Neighbors"""
        if len(data) < 50:
            return 1
        
        for dim in range(1, max_dim + 1):
            # Create time-delay embedding
            embedded = self._time_delay_embedding(data, dim)
            
            # Check for false nearest neighbors
            fnn_ratio = self._false_nearest_neighbors(embedded)
            
            if fnn_ratio < 0.1:  # Less than 10% false neighbors
                return dim
        
        return max_dim
    
    def _time_delay_embedding(self, data: np.ndarray, dim: int, tau: int = 1) -> np.ndarray:
        """Create time-delay embedding"""
        n = len(data) - (dim - 1) * tau
        embedded = np.zeros((n, dim))
        
        for i in range(dim):
            embedded[:, i] = data[i * tau:i * tau + n]
        
        return embedded
    
    def _false_nearest_neighbors(self, embedded: np.ndarray, threshold: float = 15.0) -> float:
        """Compute false nearest neighbors ratio"""
        n_points, dim = embedded.shape
        
        if n_points < 10:
            return 1.0
        
        false_neighbors = 0
        total_neighbors = 0
        
        # Sample subset for efficiency
        sample_size = min(50, n_points)
        indices = np.random.choice(n_points, sample_size, replace=False)
        
        for i in indices:
            # Find nearest neighbor in current dimension
            distances = np.sum((embedded - embedded[i]) ** 2, axis=1)
            distances[i] = np.inf  # Exclude self
            nearest_idx = np.argmin(distances)
            
            # Check if it remains nearest in higher dimension
            if dim > 1:
                dist_current = distances[nearest_idx]
                
                # Simple check: if distance increases dramatically, it's false
                if dist_current > 0:
                    relative_increase = np.sqrt(distances[nearest_idx]) / np.sqrt(dist_current)
                    if relative_increase > threshold:
                        false_neighbors += 1
                
                total_neighbors += 1
        
        return false_neighbors / max(1, total_neighbors)
    
    def _count_connected_components(self, adjacency: np.ndarray) -> int:
        """Count connected components in graph"""
        n = len(adjacency)
        visited = np.zeros(n, dtype=bool)
        components = 0
        
        def dfs(node):
            visited[node] = True
            for neighbor in range(n):
                if adjacency[node, neighbor] and not visited[neighbor]:
                    dfs(neighbor)
        
        for i in range(n):
            if not visited[i]:
                dfs(i)
                components += 1
        
        return components
    
    def _coarse_grain(self, data: np.ndarray, scale: int) -> np.ndarray:
        """Coarse-grain time series for multiscale analysis"""
        n = len(data) // scale
        coarse = np.zeros(n)
        
        for i in range(n):
            coarse[i] = np.mean(data[i*scale:(i+1)*scale])
        
        return coarse
    
    def _sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Compute sample entropy"""
        N = len(data)
        if N < m + 1:
            return 0.0
        
        r = r * np.std(data)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([data[i:i+m] for i in range(N - m + 1)])
            phi = 0
            
            for i in range(len(patterns) - 1):
                template = patterns[i]
                matches = sum([1 for j in range(i+1, len(patterns)) 
                             if _maxdist(template, patterns[j], m) <= r])
                if matches > 0:
                    phi += matches
            
            return phi / (N - m)
        
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        
        if phi_m == 0 or phi_m1 == 0:
            return 0.0
        
        return np.log(phi_m / phi_m1)
    
    def _fit_entropy_slope(self, scales: List[int], entropies: List[float]) -> float:
        """Fit slope of entropy vs scale"""
        if len(scales) < 2:
            return 0.0
        
        log_scales = np.log(scales)
        slope, _ = np.polyfit(log_scales, entropies, 1)
        return slope

class UniversalCollapsePredictor:
    """Advanced prediction system with ensemble methods"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.feature_extractor = EnhancedFeatureExtractor(config)
        
    def predict_collapse_probability(self, data: np.ndarray, timestamps: np.ndarray) -> float:
        """Predict collapse probability using ensemble of methods"""
        
        # Extract comprehensive features
        features = self._extract_all_features(data, timestamps)
        
        # Multiple prediction methods
        predictions = []
        
        # Method 1: Spectral acceleration threshold
        spectral_prob = self._spectral_prediction(features)
        predictions.append(spectral_prob)
        
        # Method 2: Topological indicators
        topo_prob = self._topological_prediction(features)
        predictions.append(topo_prob)
        
        # Method 3: Quantum friction anomalies
        quantum_prob = self._quantum_prediction(features)
        predictions.append(quantum_prob)
        
        # Method 4: Entropy dynamics
        entropy_prob = self._entropy_prediction(features)
        predictions.append(entropy_prob)
        
        # Ensemble prediction with adaptive weights
        weights = self._compute_adaptive_weights(features)
        final_probability = np.average(predictions, weights=weights)
        
        return min(1.0, max(0.0, final_probability))
    
    def extract_universal_signature(self, data: np.ndarray, timestamps: np.ndarray,
                                  domain: str = 'general') -> UniversalSignature:
        """Extract complete universal collapse signature"""
        
        features = self._extract_all_features(data, timestamps)
        
        # Fit multi-phase sigmoid model
        phases = self._fit_multiphase_model(data, timestamps)
        
        # Extract universal exponents
        exponents = self._extract_universal_exponents(data, timestamps, phases)
        
        # Compute intervention metrics
        intervention_window = self._compute_intervention_window(data, features)
        intervention_type = self._select_intervention_type(features)
        
        # Estimate uncertainties
        model_confidence = self._compute_model_confidence(data, phases)
        prediction_uncertainty = self._compute_prediction_uncertainty(features)
        
        signature = UniversalSignature(
            timestamp=datetime.now(),
            domain=domain,
            
            # Spectral components
            phase_count=len(phases),
            critical_times=[p['critical_time'] for p in phases],
            steepness_ratios=[p['steepness'] for p in phases],
            saturation_pattern=features.get('saturation_pattern', []),
            
            # Topological invariants
            dimensional_reduction_factor=features.get('dimensional_reduction', 1.0),
            eigenmode_collapse_rate=features.get('eigenmode_collapse_rate', 0.0),
            topology_shift_indicator=features.get('topology_shift', 0.0),
            
            # Entropy dynamics
            entropy_production_rate=features.get('entropy_production_rate', 0.0),
            entropy_reversal_point=features.get('entropy_reversal_point'),
            
            # Universal exponents
            alpha_exponent=exponents.get('alpha', 0.0),
            beta_exponent=exponents.get('beta', 0.0),
            gamma_exponent=exponents.get('gamma', 0.0),
            
            # Quantum friction effects
            charge_tunneling_rate=features.get('mean_tunneling_rate', 0.0),
            fractal_dimension=features.get('fractal_dimension', 1.0),
            quantum_friction_coeff=features.get('quantum_friction_coeff', 0.0),
            
            # Intervention metrics
            intervention_window=intervention_window,
            intervention_priority=self.predict_collapse_probability(data, timestamps),
            optimal_intervention_type=intervention_type,
            
            # Confidence metrics
            model_confidence=model_confidence,
            prediction_uncertainty=prediction_uncertainty,
            
            metadata={
                'data_length': len(data),
                'feature_count': len(features),
                'processing_time': datetime.now()
            }
        )
        
        return signature
    
    def _extract_all_features(self, data: np.ndarray, timestamps: np.ndarray) -> Dict:
        """Extract comprehensive feature set"""
        features = {}
        
        # Basic statistical features
        features.update(self._basic_statistical_features(data))
        
        # Spectral features
        features.update(self.feature_extractor.extract_spectral_features(data, timestamps))
        
        # Topological features
        features.update(self.feature_extractor.extract_topological_features(data))
        
        # Quantum features
        features.update(self.feature_extractor.extract_quantum_features(data))
        
        # Multiscale entropy
        features.update(self.feature_extractor.extract_multiscale_entropy(data))
        
        # Wavelet features
        features.update(self._wavelet_features(data))
        
        return features
    
    def _basic_statistical_features(self, data: np.ndarray) -> Dict:
        """Basic statistical features"""
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'skewness': self._skewness(data),
            'kurtosis': self._kurtosis(data),
            'range': np.max(data) - np.min(data),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25)
        }
    
    def _wavelet_features(self, data: np.ndarray) -> Dict:
        """Wavelet-based features"""
        features = {}
        
        # Continuous wavelet transform
        scales = np.arange(1, 32)
        coeffs, _ = pywt.cwt(data, scales, 'morl')
        
        features['wavelet_energy'] = np.sum(coeffs**2, axis=1).tolist()
        features['wavelet_entropy'] = scipy_entropy(np.sum(coeffs**2, axis=1))
        
        # Discrete wavelet transform
        coeffs_dwt = pywt.wavedec(data, 'db4', level=4)
        features['dwt_energy_ratio'] = [np.sum(c**2) for c in coeffs_dwt]
        
        return features
    
    def _spectral_prediction(self, features: Dict) -> float:
        """Prediction based on spectral acceleration"""
        max_accel = features.get('max_acceleration', 0)
        accel_peaks = features.get('acceleration_peaks', 0)
        saturation_rate = features.get('saturation_rate', 0)
        
        # Normalize and combine
        accel_score = min(1.0, max_accel / 10.0)  # Assume threshold of 10
        peaks_score = min(1.0, accel_peaks / 5.0)  # Max 5 peaks
        saturation_score = min(1.0, saturation_rate * 2.0)
        
        return 0.4 * accel_score + 0.3 * peaks_score + 0.3 * saturation_score
    
    def _topological_prediction(self, features: Dict) -> float:
        """Prediction based on topological changes"""
        betti_0 = features.get('betti_0', 5)  # Default to 5 components
        betti_1 = features.get('betti_1', 0)
        embedding_dim = features.get('embedding_dimension', 3)
        
        # Low connectivity suggests collapse
        connectivity_score = max(0, 1.0 - betti_0 / 10.0)
        
        # High embedding dimension suggests complexity
        complexity_score = min(1.0, embedding_dim / 5.0)
        
        return 0.6 * connectivity_score + 0.4 * complexity_score
    
    def _quantum_prediction(self, features: Dict) -> float:
        """Prediction based on quantum friction"""
        tunneling_rate = features.get('mean_tunneling_rate', 0)
        fractal_dim = features.get('fractal_dimension', 1.5)
        coherence = features.get('quantum_coherence', 1.0)
        
        # High tunneling suggests instability
        tunneling_score = min(1.0, tunneling_rate * 10.0)
        
        # Fractal dimension effects
        fractal_score = abs(fractal_dim - 1.5) / 1.5  # Deviation from typical
        
        # Low coherence suggests disorder
        coherence_score = max(0, 1.0 - coherence / 2.0)
        
        return 0.4 * tunneling_score + 0.3 * fractal_score + 0.3 * coherence_score
    
    def _entropy_prediction(self, features: Dict) -> float:
        """Prediction based on entropy dynamics"""
        multiscale_entropy = features.get('multiscale_entropy_mean', 1.0)
        entropy_slope = features.get('multiscale_entropy_slope', 0)
        entropy_complexity = features.get('entropy_complexity', 0)
        
        # Entropy anomalies
        entropy_score = abs(multiscale_entropy - 1.0)  # Deviation from expected
        slope_score = min(1.0, abs(entropy_slope))
        complexity_score = min(1.0, entropy_complexity / 0.5)
        
        return 0.4 * entropy_score + 0.3 * slope_score + 0.3 * complexity_score
    
    def _compute_adaptive_weights(self, features: Dict) -> np.ndarray:
        """Compute adaptive weights for ensemble prediction"""
        # Base weights
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Adjust based on feature quality
        if features.get('acceleration_peaks', 0) > 0:
            weights[0] *= 1.2  # Boost spectral if peaks detected
        
        if features.get('betti_0', 5) < 3:
            weights[1] *= 1.3  # Boost topological if low connectivity
        
        if features.get('fractal_dimension', 1.5) > 2.0:
            weights[2] *= 1.1  # Boost quantum if high fractal dimension
        
        # Normalize
        return weights / np.sum(weights)
    
    def _fit_multiphase_model(self, data: np.ndarray, timestamps: np.ndarray) -> List[Dict]:
        """Fit multi-phase sigmoid model"""
        phases = []
        
        # Find significant gradient changes
        gradient = np.gradient(data)
        peaks, _ = signal.find_peaks(np.abs(gradient), height=np.std(gradient))
        
        if len(peaks) == 0:
            # Single phase model
            phases.append({
                'weight': np.max(data) - np.min(data),
                'steepness': 1.0,
                'critical_time': timestamps[len(timestamps)//2]
            })
        else:
            # Multi-phase based on peaks
            for peak in peaks:
                phases.append({
                    'weight': np.abs(gradient[peak]),
                    'steepness': np.abs(np.gradient(gradient)[peak]),
                    'critical_time': timestamps[peak]
                })
        
        return phases
    
    def _extract_universal_exponents(self, data: np.ndarray, timestamps: np.ndarray,
                                   phases: List[Dict]) -> Dict:
        """Extract universal scaling exponents"""
        exponents = {}
        
        if not phases:
            return exponents
        
        # Use first critical point
        tau_c = phases[0]['critical_time']
        tau_c_idx = np.argmin(np.abs(timestamps - tau_c))
        
        if tau_c_idx > 10:
            # Extract data before critical point
            t_before = timestamps[:tau_c_idx]
            y_before = data[:tau_c_idx]
            
            # Power law fitting: y ~ (tau_c - t)^alpha
            tau_diff = tau_c - t_before
            tau_diff = tau_diff[tau_diff > 0]
            y_subset = y_before[-len(tau_diff):]
            
            if len(tau_diff) > 5:
                try:
                    log_tau = np.log(tau_diff)
                    log_y = np.log(np.abs(y_subset - np.mean(y_subset)) + 1e-10)
                    
                    alpha, _ = np.polyfit(log_tau, log_y, 1)
                    exponents['alpha'] = alpha
                except:
                    exponents['alpha'] = 0.0
        
        # Additional exponents (simplified)
        exponents['beta'] = exponents.get('alpha', 0.0) * 0.5  # Typical relation
        exponents['gamma'] = exponents.get('alpha', 0.0) * 2.0  # Typical relation
        
        return exponents
    
    def _compute_intervention_window(self, data: np.ndarray, features: Dict) -> Tuple[float, float]:
        """Compute optimal intervention window"""
        n = len(data)
        
        # Based on acceleration peaks
        if features.get('acceleration_peaks', 0) > 0:
            # Find first significant acceleration
            accel = np.abs(np.gradient(np.gradient(data)))
            threshold = np.percentile(accel, 80)
            candidates = np.where(accel > threshold)[0]
            
            if len(candidates) > 0:
                start_idx = max(0, candidates[0] - 10)
                end_idx = min(n-1, candidates[0] + 5)
                return (start_idx, end_idx)
        
        # Default: before middle
        return (n//3, n//2)
    
    def _select_intervention_type(self, features: Dict) -> str:
        """Select optimal intervention type"""
        kurtosis = features.get('kurtosis', 0)
        entropy = features.get('multiscale_entropy_mean', 1.0)
        connectivity = features.get('betti_0', 5)
        tunneling = features.get('mean_tunneling_rate', 0)
        
        if kurtosis > 5:
            return 'pressure_relief'
        elif entropy < 0.5:
            return 'entropy_injection'
        elif connectivity <