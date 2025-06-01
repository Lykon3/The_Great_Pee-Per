# config_driven_collapse_system.py
"""
Config-Driven Universal Collapse Control System
==============================================
A modular, configuration-based implementation of the unified collapse detection
and intervention framework with support for multiple domains and methods.
"""

import numpy as np
import pandas as pd
import yaml
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# Scientific computing
from scipy import signal, optimize, interpolate
from scipy.special import gamma
import pywt

# Machine learning
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import DBSCAN

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration structure
DEFAULT_CONFIG = """
# Universal Collapse Control System Configuration
# =============================================

system:
  name: "Universal Collapse Control System"
  version: "1.0.0"
  log_level: "INFO"
  output_dir: "./outputs"
  
models:
  oscillatory:
    enabled: true
    type: "chirp"  # Options: simple, chirp, multi_phase
    max_oscillations: 5
    damping_bounds: [0.1, 10.0]
    chirp_rate_bounds: [0.001, 0.1]
    
  fractional:
    enabled: true
    alpha_range: [0.1, 0.9]
    memory_depth: 100
    method: "grunwald_letnikov"  # Options: grunwald_letnikov, caputo
    
  sigmoid:
    enabled: true
    phases: [1, 2, 3]  # Number of phases to try
    optimization_method: "differential_evolution"
    
  gpr:
    enabled: false  # Computationally expensive
    kernel: "rbf"
    length_scale_bounds: [0.1, 100.0]
    
features:
  time_domain:
    enabled: true
    compute_moments: true
    compute_percentiles: true
    
  entropy:
    enabled: true
    methods: ["shannon", "sample", "approximate", "multiscale"]
    window_size: 50
    overlap: 0.5
    bins: 10
    
  frequency_domain:
    enabled: true
    methods: ["fft", "welch", "multitaper"]
    nperseg: 256
    
  wavelet:
    enabled: true
    wavelet: "morlet"  # Options: morlet, db4, coif2
    scales: [1, 2, 4, 8, 16, 32]
    
  decomposition:
    enabled: true
    method: "emd"  # Options: emd, vmd, both
    max_imfs: 10
    vmd_modes: 5
    
  topology:
    enabled: true
    compute_betti: true
    persistence_threshold: 0.01
    max_dimension: 2
    
  manifold:
    enabled: true
    method: "umap"  # Options: umap, tsne, diffusion_map
    n_neighbors: 15
    min_dist: 0.1
    
streaming:
  enabled: true
  buffer_size: 1000
  update_interval: 0.1  # seconds
  intervention_threshold: 0.75
  
  intervention_types:
    pressure_relief:
      trigger: "high_kurtosis"
      threshold: 10.0
    entropy_injection:
      trigger: "low_entropy"
      threshold: 0.1
    topology_restructuring:
      trigger: "low_connectivity"
      threshold: 2
    adaptive_damping:
      trigger: "default"
      
domain_specific:
  financial:
    orderbook_depth: 10
    tick_aggregation: "1s"
    liquidity_threshold: 0.2
    
  plasma:
    mhd_modes: [1, 2, 3]
    q_profile_resolution: 100
    disruption_precursor_time: 0.1  # seconds
    
  neural:
    eeg_channels: 64
    frequency_bands:
      delta: [0.5, 4]
      theta: [4, 8]
      alpha: [8, 13]
      beta: [13, 30]
      gamma: [30, 100]
    seizure_prediction_horizon: 120  # seconds
    
  cosmic:
    redshift_bins: 20
    halo_mass_threshold: 1e12  # solar masses
    filament_threshold: 2.0  # overdensity
    
output:
  save_signatures: true
  save_interventions: true
  save_visualizations: true
  formats: ["json", "csv", "npz"]
  compression: true
  
visualization:
  plot_style: "seaborn"
  figure_size: [12, 8]
  dpi: 150
  save_formats: ["png", "pdf"]
  
validation:
  cross_validation_folds: 5
  bootstrap_samples: 1000
  confidence_level: 0.95
  synthetic_test_cases: ["lorenz", "rossler", "plasma_turbulence", "market_crash"]
"""

@dataclass
class CollapseSignature:
    """Data class for collapse signatures"""
    timestamp: datetime
    domain: str
    collapse_type: str
    phase_count: int
    critical_times: List[float]
    universal_exponents: Dict[str, float]
    intervention_window: Tuple[float, float]
    confidence: float
    features: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return asdict(self)

@dataclass
class Intervention:
    """Data class for interventions"""
    timestamp: datetime
    intervention_type: str
    trigger_features: Dict[str, float]
    collapse_probability: float
    success: Optional[bool] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return asdict(self)

class BaseModel(ABC):
    """Abstract base class for collapse models"""
    
    @abstractmethod
    def fit(self, t: np.ndarray, data: np.ndarray) -> Dict[str, Any]:
        """Fit model to data"""
        pass
    
    @abstractmethod
    def predict(self, t: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Predict using fitted parameters"""
        pass
    
    @abstractmethod
    def compute_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R-squared value"""
        pass

class OscillatoryChirpModel(BaseModel):
    """Enhanced oscillatory model with frequency chirp"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def model_function(self, t, A_base, k, t_c, A_osc, gamma, omega_0, beta, phi):
        """Chirped oscillatory collapse model"""
        sigmoid = A_base / (1 + np.exp(-k * (t - t_c)))
        phase = omega_0 * (t - t_c) + beta * (t - t_c)**2 + phi
        oscillation = A_osc * np.exp(-gamma * np.abs(t - t_c)) * np.cos(phase)
        window = np.exp(-((t - t_c) / (2 * gamma))**2)
        return sigmoid + oscillation * window
    
    def fit(self, t: np.ndarray, data: np.ndarray) -> Dict[str, Any]:
        """Fit chirped oscillatory model"""
        try:
            # Initial guess
            t_mid = t[len(t)//2]
            p0 = [np.max(data), 1.0, t_mid, 0.2, 1.0, 1.0, 0.01, 0]
            
            # Bounds
            bounds = (
                [0, 0.1, t[0], 0, 0.1, 0.1, 0.001, -np.pi],
                [2*np.max(data), 10, t[-1], 1, 10, 10, 0.1, np.pi]
            )
            
            popt, pcov = optimize.curve_fit(
                self.model_function, t, data, 
                p0=p0, bounds=bounds, maxfev=5000
            )
            
            y_pred = self.model_function(t, *popt)
            r2 = self.compute_r2(data, y_pred)
            
            return {
                'params': popt,
                'covariance': pcov,
                'r2': r2,
                'fitted_curve': y_pred,
                'model_type': 'oscillatory_chirp'
            }
            
        except Exception as e:
            logger.error(f"Oscillatory chirp fit failed: {e}")
            return None
    
    def predict(self, t: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Predict using fitted parameters"""
        return self.model_function(t, *params['params'])
    
    def compute_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R-squared"""
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

class ConfigurableCollapseSystem:
    """Main system class with full configuration support"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with configuration"""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._initialize_models()
        self._initialize_storage()
        
        # State tracking
        self.state_history = []
        self.signatures = []
        self.interventions = []
        
        logger.info(f"Initialized {self.config['system']['name']} v{self.config['system']['version']}")
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file or use defaults"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
        else:
            config = yaml.safe_load(DEFAULT_CONFIG)
            logger.info("Using default configuration")
        
        return config
    
    def _setup_logging(self):
        """Configure logging based on config"""
        log_level = getattr(logging, self.config['system']['log_level'].upper())
        logging.getLogger().setLevel(log_level)
        
        # Create output directory
        self.output_dir = Path(self.config['system']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add file handler
        file_handler = logging.FileHandler(
            self.output_dir / f"collapse_system_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
        file_handler.setLevel(log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def _initialize_models(self):
        """Initialize enabled models"""
        self.models = {}
        
        if self.config['models']['oscillatory']['enabled']:
            self.models['oscillatory'] = OscillatoryChirpModel(
                self.config['models']['oscillatory']
            )
            logger.info("Initialized oscillatory chirp model")
        
        # Add other models as implemented
        # self.models['fractional'] = FractionalDerivativeModel(...)
        # self.models['sigmoid'] = MultiPhaseSigmoidModel(...)
        
    def _initialize_storage(self):
        """Initialize data storage"""
        self.storage_dir = self.output_dir / "data"
        self.storage_dir.mkdir(exist_ok=True)
        
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
    
    def compute_emd(self, signal: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Empirical Mode Decomposition with config support"""
        if not self.config['features']['decomposition']['enabled']:
            return [], signal
        
        method = self.config['features']['decomposition']['method']
        
        if method in ['emd', 'both']:
            # EMD implementation (simplified)
            imfs = []
            residual = signal.copy()
            max_imfs = self.config['features']['decomposition']['max_imfs']
            
            for _ in range(max_imfs):
                # Simplified sifting process
                imf = self._sift_imf(residual)
                if imf is None:
                    break
                imfs.append(imf)
                residual -= imf
            
            return imfs, residual
        
        return [], signal
    
    def _sift_imf(self, signal: np.ndarray) -> Optional[np.ndarray]:
        """Simplified IMF extraction"""
        # This is a placeholder - full implementation would be more complex
        if len(signal) < 10:
            return None
        
        # Find peaks and troughs
        peaks, _ = signal.find_peaks()
        troughs, _ = signal.find_peaks(-signal)
        
        if len(peaks) < 3 or len(troughs) < 3:
            return None
        
        # Simplified envelope interpolation
        x = np.arange(len(signal))
        upper = np.interp(x, peaks, signal[peaks])
        lower = np.interp(x, troughs, signal[troughs])
        mean_env = (upper + lower) / 2
        
        return signal - mean_env
    
    def compute_entropy_features(self, data: np.ndarray) -> Dict[str, float]:
        """Compute various entropy measures based on config"""
        features = {}
        
        if not self.config['features']['entropy']['enabled']:
            return features
        
        methods = self.config['features']['entropy']['methods']
        
        if 'shannon' in methods:
            bins = self.config['features']['entropy']['bins']
            hist, _ = np.histogram(data, bins=bins)
            hist = hist / np.sum(hist)
            hist = hist[hist > 0]
            features['shannon_entropy'] = -np.sum(hist * np.log2(hist))
        
        if 'sample' in methods:
            # Simplified sample entropy
            features['sample_entropy'] = self._sample_entropy(data)
        
        return features
    
    def _sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Compute sample entropy"""
        N = len(data)
        r = r * np.std(data)
        
        def maxdist(xi, xj):
            return max(abs(float(a) - float(b)) for a, b in zip(xi, xj))
        
        def phi(m):
            patterns = [data[i:i+m] for i in range(N - m + 1)]
            C = []
            
            for i, pattern_i in enumerate(patterns):
                matches = sum(1 for j, pattern_j in enumerate(patterns) 
                            if i != j and maxdist(pattern_i, pattern_j) <= r)
                if matches > 0:
                    C.append(matches / (N - m))
            
            return np.mean(C) if C else 0
        
        phi_m = phi(m)
        phi_m1 = phi(m + 1)
        
        return -np.log(phi_m1 / phi_m) if phi_m > 0 and phi_m1 > 0 else 0
    
    def compute_manifold_features(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute manifold learning features"""
        features = {}
        
        if not self.config['features']['manifold']['enabled']:
            return features
        
        method = self.config['features']['manifold']['method']
        
        if method == 'umap' and len(data) > 50:
            try:
                # Reshape for UMAP
                X = data.reshape(-1, 1)
                
                reducer = umap.UMAP(
                    n_neighbors=min(15, len(data)//4),
                    min_dist=self.config['features']['manifold']['min_dist'],
                    n_components=2,
                    random_state=42
                )
                
                embedding = reducer.fit_transform(X)
                
                features['umap_embedding'] = embedding
                features['umap_spread'] = np.std(embedding, axis=0)
                
            except Exception as e:
                logger.warning(f"UMAP computation failed: {e}")
        
        return features
    
    def extract_comprehensive_features(self, data: np.ndarray) -> Dict[str, Any]:
        """Extract all enabled features"""
        features = {}
        
        # Time domain features
        if self.config['features']['time_domain']['enabled']:
            features.update({
                'mean': np.mean(data),
                'std': np.std(data),
                'skewness': self._skewness(data),
                'kurtosis': self._kurtosis(data),
                'max': np.max(data),
                'min': np.min(data)
            })
        
        # Entropy features
        features.update(self.compute_entropy_features(data))
        
        # Wavelet features
        if self.config['features']['wavelet']['enabled']:
            wavelet = self.config['features']['wavelet']['wavelet']
            scales = self.config['features']['wavelet']['scales']
            
            coeffs = pywt.cwt(data, scales, wavelet)[0]
            features['wavelet_energy'] = np.sum(coeffs**2, axis=1).tolist()
        
        # Decomposition features
        if self.config['features']['decomposition']['enabled']:
            imfs, residual = self.compute_emd(data)
            features['num_imfs'] = len(imfs)
            features['imf_energies'] = [np.sum(imf**2) for imf in imfs]
        
        # Manifold features
        features.update(self.compute_manifold_features(data))
        
        return features
    
    def _skewness(self, data: np.ndarray) -> float:
        """Compute skewness"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3) if std > 0 else 0
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3 if std > 0 else 0
    
    def process_signal(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None,
                      domain: str = 'general') -> CollapseSignature:
        """Process a complete signal and extract collapse signature"""
        logger.info(f"Processing signal from domain: {domain}")
        
        if timestamps is None:
            timestamps = np.arange(len(data))
        
        # Normalize data
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        # Fit models
        model_fits = {}
        for model_name, model in self.models.items():
            fit_result = model.fit(timestamps, data_norm)
            if fit_result:
                model_fits[model_name] = fit_result
                logger.info(f"{model_name} fit R²: {fit_result['r2']:.4f}")
        
        # Select best model
        if model_fits:
            best_model = max(model_fits.items(), key=lambda x: x[1]['r2'])
            best_name, best_fit = best_model
        else:
            logger.warning("No models successfully fitted")
            best_name, best_fit = 'none', {}
        
        # Extract features
        features = self.extract_comprehensive_features(data)
        
        # Determine collapse characteristics
        collapse_type = self._classify_collapse_type(features)
        phase_count = self._estimate_phase_count(data_norm)
        critical_times = self._find_critical_times(data_norm, timestamps)
        intervention_window = self._compute_intervention_window(data_norm)
        
        # Create signature
        signature = CollapseSignature(
            timestamp=datetime.now(),
            domain=domain,
            collapse_type=collapse_type,
            phase_count=phase_count,
            critical_times=critical_times,
            universal_exponents=self._extract_exponents(data_norm, critical_times),
            intervention_window=intervention_window,
            confidence=best_fit.get('r2', 0.0),
            features=features,
            metadata={
                'best_model': best_name,
                'model_params': best_fit.get('params', []).tolist() if 'params' in best_fit else [],
                'data_length': len(data),
                'config_version': self.config['system']['version']
            }
        )
        
        # Store signature
        self.signatures.append(signature)
        
        # Save if configured
        if self.config['output']['save_signatures']:
            self._save_signature(signature)
        
        return signature
    
    def _classify_collapse_type(self, features: Dict[str, Any]) -> str:
        """Classify collapse type based on features"""
        kurtosis = features.get('kurtosis', 0)
        entropy = features.get('shannon_entropy', 1)
        num_imfs = features.get('num_imfs', 0)
        
        if kurtosis > 8:
            return 'sharp_transition'
        elif entropy < 0.5:
            return 'ordered_emergence'
        elif num_imfs > 5:
            return 'multi_scale_cascade'
        else:
            return 'gradual_evolution'
    
    def _estimate_phase_count(self, data: np.ndarray) -> int:
        """Estimate number of collapse phases"""
        # Simplified - look for inflection points
        second_deriv = np.gradient(np.gradient(data))
        peaks, _ = signal.find_peaks(np.abs(second_deriv), height=np.std(second_deriv))
        return max(1, min(len(peaks), 5))
    
    def _find_critical_times(self, data: np.ndarray, timestamps: np.ndarray) -> List[float]:
        """Find critical transition times"""
        gradient = np.gradient(data)
        peaks, _ = signal.find_peaks(np.abs(gradient), height=np.max(np.abs(gradient))*0.3)
        
        if len(peaks) > 0:
            return timestamps[peaks].tolist()
        else:
            return [timestamps[len(timestamps)//2]]
    
    def _compute_intervention_window(self, data: np.ndarray) -> Tuple[float, float]:
        """Compute optimal intervention window"""
        gradient = np.gradient(data)
        accel = np.gradient(gradient)
        
        # Find where acceleration starts increasing
        accel_smooth = signal.savgol_filter(np.abs(accel), 
                                           min(51, len(accel)//4*2+1), 3)
        
        threshold = np.percentile(accel_smooth, 70)
        intervention_start = np.where(accel_smooth > threshold)[0]
        
        if len(intervention_start) > 0:
            start = intervention_start[0]
            end = min(start + len(data)//10, len(data)-1)
            return (start, end)
        else:
            return (len(data)//3, 2*len(data)//3)
    
    def _extract_exponents(self, data: np.ndarray, critical_times: List[float]) -> Dict[str, float]:
        """Extract universal scaling exponents"""
        exponents = {}
        
        if not critical_times:
            return exponents
        
        # Simplified power-law fitting near first critical point
        tc_idx = int(critical_times[0])
        if tc_idx > 10:
            t = np.arange(tc_idx)
            y = data[:tc_idx]
            
            # Avoid log of zero
            y_shifted = np.abs(1 - y/np.max(y)) + 1e-10
            
            try:
                log_t = np.log(t[1:] + 1)
                log_y = np.log(y_shifted[1:])
                alpha, _ = np.polyfit(log_t, log_y, 1)
                exponents['alpha'] = float(alpha)
            except:
                exponents['alpha'] = None
        
        return exponents
    
    def _save_signature(self, signature: CollapseSignature):
        """Save signature to configured formats"""
        timestamp_str = signature.timestamp.strftime("%Y%m%d_%H%M%S")
        base_name = f"signature_{signature.domain}_{timestamp_str}"
        
        formats = self.config['output']['formats']
        
        if 'json' in formats:
            json_path = self.storage_dir / f"{base_name}.json"
            with open(json_path, 'w') as f:
                json.dump(signature.to_dict(), f, indent=2, default=str)
            logger.info(f"Saved signature to {json_path}")
        
        if 'csv' in formats:
            # Flatten features for CSV
            flat_data = {
                'timestamp': signature.timestamp,
                'domain': signature.domain,
                'collapse_type': signature.collapse_type,
                'phase_count': signature.phase_count,
                'confidence': signature.confidence
            }
            flat_data.update({f"feature_{k}": v for k, v in signature.features.items() 
                            if isinstance(v, (int, float, str))})
            
            df = pd.DataFrame([flat_data])
            csv_path = self.storage_dir / f"{base_name}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved signature to {csv_path}")
    
    def stream_process(self, data_generator, domain: str = 'general'):
        """Process streaming data with intervention capabilities"""
        logger.info(f"Starting streaming analysis for domain: {domain}")
        
        buffer = []
        timestamps = []
        intervention_active = False
        
        for timestamp, value in data_generator:
            buffer.append(value)
            timestamps.append(timestamp)
            
            # Maintain buffer size
            if len(buffer) > self.config['streaming']['buffer_size']:
                buffer.pop(0)
                timestamps.pop(0)
            
            # Process when enough data accumulated
            if len(buffer) >= 50:
                features = self.extract_comprehensive_features(np.array(buffer))
                
                # Predict collapse probability
                collapse_prob = self._predict_collapse_probability(features)
                
                # Check intervention threshold
                if (collapse_prob > self.config['streaming']['intervention_threshold'] 
                    and not intervention_active):
                    
                    intervention_type = self._select_intervention(features)
                    
                    intervention = Intervention(
                        timestamp=datetime.now(),
                        intervention_type=intervention_type,
                        trigger_features=features,
                        collapse_probability=collapse_prob,
                        metadata={'domain': domain, 'buffer_size': len(buffer)}
                    )
                    
                    self.interventions.append(intervention)
                    intervention_active = True
                    
                    logger.warning(f"🚨 INTERVENTION TRIGGERED: {intervention_type} "
                                 f"(probability: {collapse_prob:.2%})")
                    
                    if self.config['output']['save_interventions']:
                        self._save_intervention(intervention)
                    
                    yield intervention
                
                # Reset intervention flag if probability drops
                if collapse_prob < 0.5:
                    intervention_active = False
    
    def _predict_collapse_probability(self, features: Dict[str, Any]) -> float:
        """Predict collapse probability from features"""
        # Simplified heuristic - replace with trained model
        risk_score = 0.0
        
        # Check each intervention trigger
        for int_type, int_config in self.config['streaming']['intervention_types'].items():
            trigger = int_config['trigger']
            threshold = int_config.get('threshold', 0)
            
            if trigger == 'high_kurtosis' and features.get('kurtosis', 0) > threshold:
                risk_score += 0.3
            elif trigger == 'low_entropy' and features.get('shannon_entropy', 1) < threshold:
                risk_score += 0.3
            elif trigger == 'low_connectivity' and features.get('num_imfs', 10) < threshold:
                risk_score += 0.2
        
        return min(risk_score, 1.0)
    
    def _select_intervention(self, features: Dict[str, Any]) -> str:
        """Select appropriate intervention type"""
        for int_type, int_config in self.config['streaming']['intervention_types'].items():
            trigger = int_config['trigger']
            threshold = int_config.get('threshold', 0)
            
            if trigger == 'high_kurtosis' and features.get('kurtosis', 0) > threshold:
                return int_type
            elif trigger == 'low_entropy' and features.get('shannon_entropy', 1) < threshold:
                return int_type
            elif trigger == 'low_connectivity' and features.get('num_imfs', 10) < threshold:
                return int_type
        
        return 'adaptive_damping'
    
    def _save_intervention(self, intervention: Intervention):
        """Save intervention record"""
        timestamp_str = intervention.timestamp.strftime("%Y%m%d_%H%M%S")
        json_path = self.storage_dir / f"intervention_{timestamp_str}.json"
        
        with open(json