import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import pandas as pd

class UniversalCollapseExtractor:
    """Extract universal collapse signatures from spectral data"""
    
    def __init__(self, spectral_data, time_axis):
        self.data = spectral_data
        self.time = time_axis
        self.results = {}
        
    # Single-phase sigmoid model
    def sigmoid_single(self, t, A, k, tc, offset):
        """Single-phase collapse function"""
        return A / (1 + np.exp(-k * (t - tc))) + offset
    
    # Multi-phase cascade model
    def sigmoid_multiphase(self, t, *params):
        """Multi-phase collapse with n transitions"""
        n_phases = len(params) // 3
        result = 0
        for i in range(n_phases):
            A = params[3*i]
            k = params[3*i + 1]
            tc = params[3*i + 2]
            result += A / (1 + np.exp(-k * (t - tc)))
        return result
    
    # Ricci-weighted decay model
    def ricci_collapse(self, t, A, k, tc, alpha, beta):
        """Collapse with Ricci curvature correction"""
        base = A / (1 + np.exp(-k * (t - tc)))
        ricci_factor = np.exp(-alpha * (t - tc)**2) * (1 + beta * t)
        return base * ricci_factor
    
    def compute_derivatives(self, fitted_curve):
        """Compute first and second derivatives"""
        # Use spline for smooth derivatives
        spline = UnivariateSpline(self.time, fitted_curve, s=0.1)
        first_deriv = spline.derivative(1)(self.time)
        second_deriv = spline.derivative(2)(self.time)
        return first_deriv, second_deriv
    
    def compute_acceleration_profile(self, second_deriv):
        """Extract acceleration peaks A(τ)"""
        acceleration = np.abs(second_deriv)
        peaks, properties = find_peaks(acceleration, 
                                     height=np.max(acceleration)*0.1)
        return acceleration, peaks, properties
    
    def compute_saturation_metric(self, acceleration):
        """Calculate saturation progression S(τ)"""
        cumulative = np.cumsum(acceleration)
        saturation = cumulative / np.max(cumulative)
        return saturation
    
    def fit_all_models(self):
        """Fit all collapse models and compare"""
        
        # Normalize data
        norm_data = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        
        # 1. Single-phase sigmoid
        print("Fitting single-phase sigmoid...")
        try:
            p0_single = [1, 1, np.median(self.time), 0]
            popt_single, _ = curve_fit(self.sigmoid_single, self.time, norm_data, 
                                      p0=p0_single, maxfev=5000)
            fit_single = self.sigmoid_single(self.time, *popt_single)
            r2_single = 1 - np.sum((norm_data - fit_single)**2) / np.sum((norm_data - np.mean(norm_data))**2)
            self.results['single_phase'] = {
                'params': popt_single,
                'fit': fit_single,
                'r2': r2_single
            }
        except:
            print("Single-phase fit failed")
            
        # 2. Multi-phase cascade (try 2 and 3 phases)
        for n_phases in [2, 3]:
            print(f"Fitting {n_phases}-phase cascade...")
            try:
                # Initialize parameters
                p0_multi = []
                for i in range(n_phases):
                    p0_multi.extend([1/n_phases, 1, 
                                   np.min(self.time) + i*(np.max(self.time)-np.min(self.time))/n_phases])
                
                # Use differential evolution for global optimization
                bounds = []
                for i in range(n_phases):
                    bounds.extend([(0, 1), (0.1, 10), 
                                 (np.min(self.time), np.max(self.time))])
                
                result = differential_evolution(
                    lambda p: np.sum((norm_data - self.sigmoid_multiphase(self.time, *p))**2),
                    bounds, seed=42, maxiter=1000
                )
                
                fit_multi = self.sigmoid_multiphase(self.time, *result.x)
                r2_multi = 1 - np.sum((norm_data - fit_multi)**2) / np.sum((norm_data - np.mean(norm_data))**2)
                
                self.results[f'{n_phases}_phase'] = {
                    'params': result.x,
                    'fit': fit_multi,
                    'r2': r2_multi
                }
            except:
                print(f"{n_phases}-phase fit failed")
        
        # 3. Ricci-weighted model
        print("Fitting Ricci-weighted model...")
        try:
            p0_ricci = [1, 1, np.median(self.time), 0.01, 0.01]
            popt_ricci, _ = curve_fit(self.ricci_collapse, self.time, norm_data, 
                                    p0=p0_ricci, maxfev=5000)
            fit_ricci = self.ricci_collapse(self.time, *popt_ricci)
            r2_ricci = 1 - np.sum((norm_data - fit_ricci)**2) / np.sum((norm_data - np.mean(norm_data))**2)
            self.results['ricci'] = {
                'params': popt_ricci,
                'fit': fit_ricci,
                'r2': r2_ricci
            }
        except:
            print("Ricci fit failed")
    
    def extract_universal_signature(self):
        """Extract the universal collapse signature"""
        
        # Find best model
        best_model = max(self.results.items(), 
                        key=lambda x: x[1]['r2'] if 'r2' in x[1] else -np.inf)
        model_name, model_data = best_model
        
        print(f"\nBest model: {model_name} (R² = {model_data['r2']:.4f})")
        
        # Extract universal features
        fitted_curve = model_data['fit']
        first_deriv, second_deriv = self.compute_derivatives(fitted_curve)
        acceleration, peaks, peak_props = self.compute_acceleration_profile(second_deriv)
        saturation = self.compute_saturation_metric(acceleration)
        
        # Find critical times
        critical_times = []
        if model_name == 'single_phase':
            critical_times = [model_data['params'][2]]  # tc
        elif '2_phase' in model_name:
            critical_times = [model_data['params'][2], model_data['params'][5]]
        elif '3_phase' in model_name:
            critical_times = [model_data['params'][2], model_data['params'][5], 
                            model_data['params'][8]]
        
        # Extract universal exponents
        # Find power-law regions near critical points
        universal_exponents = self.extract_scaling_exponents(fitted_curve, critical_times)
        
        signature = {
            'model_type': model_name,
            'model_quality': model_data['r2'],
            'critical_times': critical_times,
            'acceleration_peaks': self.time[peaks],
            'max_acceleration': np.max(acceleration),
            'saturation_threshold': self.time[np.where(saturation > 0.9)[0][0]] if np.any(saturation > 0.9) else None,
            'universal_exponents': universal_exponents,
            'phase_count': len(critical_times)
        }
        
        self.results['universal_signature'] = signature
        return signature
    
    def extract_scaling_exponents(self, fitted_curve, critical_times):
        """Extract universal scaling exponents near critical points"""
        exponents = {}
        
        for i, tc in enumerate(critical_times):
            # Find region around critical time
            mask = np.abs(self.time - tc) < 0.1 * (np.max(self.time) - np.min(self.time))
            if np.sum(mask) < 10:
                continue
                
            t_local = self.time[mask] - tc
            y_local = fitted_curve[mask]
            
            # Fit power law: y ~ |t - tc|^alpha
            try:
                # Only use points before transition
                pre_mask = t_local < 0
                if np.sum(pre_mask) > 5:
                    log_t = np.log(np.abs(t_local[pre_mask]))
                    log_y = np.log(np.abs(1 - y_local[pre_mask]) + 1e-10)
                    alpha = np.polyfit(log_t, log_y, 1)[0]
                    exponents[f'alpha_{i}'] = alpha
            except:
                pass
                
        return exponents
    
    def visualize_results(self):
        """Create comprehensive visualization of collapse analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: All model fits
        ax = axes[0, 0]
        norm_data = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        ax.scatter(self.time, norm_data, alpha=0.5, label='Data', s=20)
        
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for i, (name, result) in enumerate(self.results.items()):
            if 'fit' in result:
                ax.plot(self.time, result['fit'], 
                       label=f"{name} (R²={result['r2']:.3f})", 
                       linewidth=2, color=colors[i % len(colors)])
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Normalized Signal')
        ax.set_title('Collapse Model Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Best fit with derivatives
        ax = axes[0, 1]
        best_model = max(self.results.items(), 
                        key=lambda x: x[1]['r2'] if 'r2' in x[1] else -np.inf)
        model_name, model_data = best_model
        
        ax.plot(self.time, model_data['fit'], 'b-', linewidth=2, label='Best Fit')
        
        # Add derivatives
        first_deriv, second_deriv = self.compute_derivatives(model_data['fit'])
        ax2 = ax.twinx()
        ax2.plot(self.time, first_deriv, 'g--', label='1st Derivative', alpha=0.7)
        ax2.plot(self.time, second_deriv, 'r--', label='2nd Derivative', alpha=0.7)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Signal', color='b')
        ax2.set_ylabel('Derivatives', color='r')
        ax.set_title(f'Best Model: {model_name}')
        
        # Plot 3: Acceleration profile
        ax = axes[0, 2]
        acceleration, peaks, _ = self.compute_acceleration_profile(second_deriv)
        ax.plot(self.time, acceleration, 'k-', linewidth=2)
        ax.scatter(self.time[peaks], acceleration[peaks], 
                  color='red', s=100, zorder=5, label='Peaks')
        ax.set_xlabel('Time')
        ax.set_ylabel('|d²F/dτ²|')
        ax.set_title('Acceleration Profile A(τ)')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Saturation metric
        ax = axes[1, 0]
        saturation = self.compute_saturation_metric(acceleration)
        ax.plot(self.time, saturation, 'purple', linewidth=3)
        ax.axhline(y=0.9, color='red', linestyle='--', label='90% Threshold')
        ax.fill_between(self.time, 0.7, 0.9, alpha=0.3, color='yellow', 
                       label='Intervention Window')
        ax.set_xlabel('Time')
        ax.set_ylabel('S(τ)')
        ax.set_title('Saturation Metric')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Phase diagram
        ax = axes[1, 1]
        if 'universal_signature' in self.results:
            sig = self.results['universal_signature']
            
            # Create phase space plot
            phase_x = model_data['fit'][:-1]
            phase_y = first_deriv[:-1]
            
            # Color by time
            colors = plt.cm.viridis(np.linspace(0, 1, len(phase_x)))
            ax.scatter(phase_x, phase_y, c=colors, s=20, alpha=0.6)
            
            # Mark critical points
            for tc in sig['critical_times']:
                idx = np.argmin(np.abs(self.time - tc))
                ax.scatter(model_data['fit'][idx], first_deriv[idx], 
                         color='red', s=200, marker='*', 
                         edgecolor='black', linewidth=2)
            
            ax.set_xlabel('F(τ)')
            ax.set_ylabel('dF/dτ')
            ax.set_title('Phase Space Trajectory')
            ax.grid(True, alpha=0.3)
        
        # Plot 6: Universal signature summary
        ax = axes[1, 2]
        ax.axis('off')
        if 'universal_signature' in self.results:
            sig = self.results['universal_signature']
            
            summary_text = f"Universal Collapse Signature\n" + "="*30 + "\n"
            summary_text += f"Model: {sig['model_type']}\n"
            summary_text += f"Quality (R²): {sig['model_quality']:.4f}\n"
            summary_text += f"Phase Count: {sig['phase_count']}\n"
            summary_text += f"Critical Times: {[f'{t:.2f}' for t in sig['critical_times']]}\n"
            summary_text += f"Max Acceleration: {sig['max_acceleration']:.3f}\n"
            
            if sig['saturation_threshold']:
                summary_text += f"Saturation at: τ = {sig['saturation_threshold']:.2f}\n"
            
            if sig['universal_exponents']:
                summary_text += "\nScaling Exponents:\n"
                for name, value in sig['universal_exponents'].items():
                    summary_text += f"  {name}: {value:.3f}\n"
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top', 
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig

# Example usage function
def analyze_collapse_data(data, time):
    """Run complete collapse analysis on spectral data"""
    
    extractor = UniversalCollapseExtractor(data, time)
    
    # Fit all models
    print("Starting collapse curve extraction...")
    extractor.fit_all_models()
    
    # Extract universal signature
    print("\nExtracting universal signature...")
    signature = extractor.extract_universal_signature()
    
    # Visualize results
    print("\nGenerating visualizations...")
    fig = extractor.visualize_results()
    
    return extractor, signature, fig

# Generate synthetic test data
def generate_test_collapse_data(n_points=500, noise_level=0.05):
    """Generate synthetic multi-phase collapse data for testing"""
    
    t = np.linspace(0, 10, n_points)
    
    # Create multi-phase collapse
    phase1 = 0.3 / (1 + np.exp(-2 * (t - 2)))
    phase2 = 0.4 / (1 + np.exp(-3 * (t - 5)))
    phase3 = 0.3 / (1 + np.exp(-1.5 * (t - 8)))
    
    signal = phase1 + phase2 + phase3
    
    # Add noise
    noise = np.random.normal(0, noise_level, n_points)
    signal += noise
    
    return t, signal

# Run demonstration
if __name__ == "__main__":
    # Generate test data
    time, data = generate_test_collapse_data()
    
    # Run analysis
    extractor, signature, fig = analyze_collapse_data(data, time)
    
    # Print results
    print("\n" + "="*50)
    print("UNIVERSAL COLLAPSE SIGNATURE EXTRACTED")
    print("="*50)
    for key, value in signature.items():
        print(f"{key}: {value}")
    
    plt.show()