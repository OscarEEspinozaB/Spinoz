"""
Método Spinoz Aplicado a Turbulencia - Análisis Computacional
Incluye datos de F1, aerodinámica y sistemas turbulentos
Oscar Espinoza - 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate, optimize
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CLASE PRINCIPAL: ANALIZADOR SPINOZ-TURBULENCIA
# ============================================================================

@dataclass
class FlowConditions:
    """Condiciones de flujo para análisis"""
    reynolds: float
    velocity: float  # m/s
    characteristic_length: float  # m
    kinematic_viscosity: float  # m²/s
    density: float  # kg/m³
    temperature: float  # K
    
class SpinozTurbulenceAnalyzer:
    """
    Analizador principal del método Spinoz aplicado a turbulencia
    """
    
    def __init__(self):
        self.results = {}
        self.experimental_data = self._load_experimental_data()
        
    def _load_experimental_data(self) -> Dict:
        """Carga datos experimentales conocidos"""
        return {
            'kolmogorov_exponent': -5/3,
            'critical_reynolds_pipe': 2300,
            'critical_reynolds_flat_plate': 5e5,
            'von_karman_constant': 0.41,
            'kolmogorov_constant': 1.5,
            'bacterial_intermittency': 0.20
        }
    
    def calculate_spinoz(self, ratio: float) -> float:
        """
        Calcula el parámetro Spinoz básico
        s = 1 - (divisor/dividendo)
        """
        return 1 - ratio
    
    def taylor_series_expansion(self, s: float, n_terms: int = 10) -> Tuple[List[float], float]:
        """
        Expande 1/(1-s) en serie de Taylor
        """
        if abs(s) >= 1:
            raise ValueError(f"Serie de Taylor requiere |s| < 1, pero s = {s}")
        
        terms = [s**i for i in range(n_terms)]
        partial_sums = np.cumsum(terms)
        exact_value = 1/(1-s)
        
        return terms, partial_sums, exact_value
    
    def laurent_series_expansion(self, s: float, n_terms: int = 10) -> Tuple[List[float], float]:
        """
        Expande 1/(1-s) en serie de Laurent para |s| > 1
        """
        if abs(s) <= 1:
            raise ValueError(f"Serie de Laurent requiere |s| > 1, pero s = {s}")
        
        terms = [-(1/s)**(i+1) for i in range(n_terms)]
        partial_sums = np.cumsum(terms)
        exact_value = 1/(1-s)
        
        return terms, partial_sums, exact_value
    
    def kolmogorov_scales(self, Re: float, L: float, nu: float) -> Dict:
        """
        Calcula las escalas de Kolmogorov
        """
        # Estimación de la tasa de disipación
        U = Re * nu / L
        epsilon = U**3 / L  # Estimación dimensional
        
        # Escalas de Kolmogorov
        eta = (nu**3 / epsilon)**0.25  # Escala de longitud
        tau = (nu / epsilon)**0.5  # Escala de tiempo
        v = (nu * epsilon)**0.25  # Escala de velocidad
        
        # Relación con escala integral
        eta_over_L = Re**(-3/4)
        
        # Parámetro Spinoz
        s = self.calculate_spinoz(eta_over_L)
        
        return {
            'eta': eta,
            'tau': tau,
            'v': v,
            'eta_over_L': eta_over_L,
            'spinoz': s,
            'epsilon': epsilon,
            'U': U
        }
    
    def energy_cascade_model(self, s: float, epsilon: float, n_levels: int = 10) -> Dict:
        """
        Modela la cascada de energía usando Spinoz
        """
        if abs(s) < 1:
            # Cascada normal (Taylor)
            cascade_factors = [s**i for i in range(n_levels)]
            energy_distribution = epsilon * np.array(cascade_factors)
            cascade_type = "Taylor (Normal)"
        else:
            # Cascada inversa (Laurent)
            cascade_factors = [-(1/s)**(i+1) for i in range(n_levels)]
            energy_distribution = epsilon * np.array(cascade_factors)
            cascade_type = "Laurent (Inversa)"
        
        cumulative_energy = np.cumsum(energy_distribution)
        
        return {
            'cascade_type': cascade_type,
            'cascade_factors': cascade_factors,
            'energy_distribution': energy_distribution,
            'cumulative_energy': cumulative_energy,
            'total_dissipation': cumulative_energy[-1]
        }
    
    def turbulent_viscosity_spinoz(self, nu: float, s: float) -> float:
        """
        Calcula la viscosidad turbulenta modificada por Spinoz
        """
        if abs(s) < 1:
            # Régimen de Taylor
            nu_t = nu / (1 - s)
        else:
            # Régimen de Laurent (valor absoluto para evitar negativos)
            nu_t = abs(nu * s / (s - 1))
        
        return nu_t
    
    def reynolds_stress_closure(self, rho: float, u_rms: float, s: float) -> float:
        """
        Modelo de cierre para esfuerzos de Reynolds usando Spinoz
        """
        tau_standard = -rho * u_rms**2
        
        if abs(s) < 1:
            amplification = 1/(1-s)
        else:
            amplification = abs(s/(s-1))
        
        tau_spinoz = tau_standard * amplification
        
        return tau_spinoz
    
    def intermittency_correction(self, s_base: float, mu: float = 0.20) -> float:
        """
        Corrección por intermitencia (Kolmogorov 1962)
        mu: parámetro de intermitencia (típicamente 0.20)
        """
        s_intermittent = 1 - mu
        correction_factor = 1/(1 - s_intermittent)
        s_corrected = s_base * correction_factor
        
        return s_corrected, correction_factor
    
    def transition_criterion(self, Re: float) -> Dict:
        """
        Criterio de transición laminar-turbulento usando Spinoz
        """
        # Para flujo en tubería
        Re_crit = self.experimental_data['critical_reynolds_pipe']
        
        # Cálculo del Spinoz
        eta_L_ratio = Re**(-3/4)
        s = self.calculate_spinoz(eta_L_ratio)
        
        # Determinar régimen
        if Re < Re_crit:
            regime = "Laminar"
        elif Re < Re_crit * 1.3:  # Zona de transición
            regime = "Transición"
        else:
            regime = "Turbulento"
        
        return {
            'reynolds': Re,
            'spinoz': s,
            'regime': regime,
            'distance_to_critical': (Re - Re_crit) / Re_crit
        }
    
    def backscatter_model(self, epsilon: float, s: float) -> Dict:
        """
        Modelo de backscatter usando el Teorema de Reversibilidad Spinoz
        """
        # Transferencia directa (forward)
        if abs(s) < 1:
            Pi_forward = epsilon / (1 - s)
        else:
            Pi_forward = epsilon * abs(s / (s - 1))
        
        # Backscatter (Teorema de Reversibilidad: 1/c = 1-s)
        Pi_backward = epsilon * (1 - s)
        
        # Transferencia neta
        Pi_net = Pi_forward - abs(Pi_backward)
        
        # Fracción de backscatter
        if Pi_forward != 0:
            backscatter_fraction = abs(Pi_backward) / Pi_forward
        else:
            backscatter_fraction = 0
        
        return {
            'forward_transfer': Pi_forward,
            'backward_transfer': Pi_backward,
            'net_transfer': Pi_net,
            'backscatter_fraction': backscatter_fraction
        }

# ============================================================================
# APLICACIONES ESPECÍFICAS A F1
# ============================================================================

class F1AerodynamicsSpinoz:
    """
    Aplicación del método Spinoz a aerodinámica de F1
    """
    
    def __init__(self):
        self.analyzer = SpinozTurbulenceAnalyzer()
        self.f1_data = self._load_f1_data()
    
    def _load_f1_data(self) -> Dict:
        """Datos típicos de F1 basados en literatura"""
        return {
            'max_speed': 350,  # km/h
            'wind_tunnel_speed': 50,  # m/s (límite FIA)
            'model_scale': 0.6,  # 60% escala
            'car_length': 5.5,  # m
            'front_wing_chord': 0.5,  # m
            'rear_wing_chord': 0.35,  # m
            'air_density': 1.225,  # kg/m³ a nivel del mar
            'air_viscosity': 1.81e-5,  # Pa·s
            'downforce_coefficient': 3.5,  # CL típico
            'drag_coefficient': 1.0,  # CD típico
            'frontal_area': 1.5,  # m²
            'ground_clearance': 0.05,  # m
            'diffuser_angle': 10,  # grados
            'track_conditions': {
                'monaco': {'efficiency': 0.3, 'avg_speed': 160},  # km/h
                'monza': {'efficiency': 0.8, 'avg_speed': 260},   # km/h
                'silverstone': {'efficiency': 0.5, 'avg_speed': 210}  # km/h
            }
        }
    
    def wing_reynolds_analysis(self, velocity_kmh: float, chord: float, 
                              altitude: float = 0) -> Dict:
        """
        Análisis de Reynolds para alas de F1
        """
        # Conversión de velocidad
        velocity = velocity_kmh / 3.6  # m/s
        
        # Propiedades del aire (ajustadas por altitud si es necesario)
        rho = self.f1_data['air_density'] * np.exp(-altitude/8000)
        mu = self.f1_data['air_viscosity']
        nu = mu / rho
        
        # Reynolds number
        Re = velocity * chord / nu
        
        # Análisis Spinoz
        flow = FlowConditions(
            reynolds=Re,
            velocity=velocity,
            characteristic_length=chord,
            kinematic_viscosity=nu,
            density=rho,
            temperature=288.15
        )
        
        # Escalas de Kolmogorov
        scales = self.analyzer.kolmogorov_scales(Re, chord, nu)
        
        # Análisis de capa límite
        boundary_layer_thickness = 0.37 * chord / (Re**0.2)
        displacement_thickness = boundary_layer_thickness / 8
        momentum_thickness = boundary_layer_thickness / 12
        
        return {
            'reynolds': Re,
            'spinoz': scales['spinoz'],
            'boundary_layer': boundary_layer_thickness,
            'displacement_thickness': displacement_thickness,
            'momentum_thickness': momentum_thickness,
            'kolmogorov_scales': scales,
            'flow_regime': 'Turbulent' if Re > 5e5 else 'Transitional'
        }
    
    def ground_effect_analysis(self, velocity_kmh: float, 
                             ground_clearance: float) -> Dict:
        """
        Análisis del efecto suelo usando Spinoz
        """
        velocity = velocity_kmh / 3.6
        
        # Relación de escalas para efecto suelo
        clearance_to_chord = ground_clearance / self.f1_data['front_wing_chord']
        
        # Spinoz del efecto suelo
        s_ground = self.analyzer.calculate_spinoz(clearance_to_chord)
        
        # Amplificación de downforce
        if abs(s_ground) < 1:
            downforce_amplification = 1/(1-s_ground)
        else:
            downforce_amplification = 2.5  # Límite práctico
        
        # Vórtices de punta
        vortex_strength = velocity * np.sqrt(downforce_amplification)
        
        return {
            'clearance_ratio': clearance_to_chord,
            'spinoz_ground': s_ground,
            'downforce_amplification': downforce_amplification,
            'vortex_strength': vortex_strength,
            'effective_downforce': self.f1_data['downforce_coefficient'] * downforce_amplification
        }
    
    def diffuser_performance(self, velocity_kmh: float, angle_deg: float) -> Dict:
        """
        Análisis del difusor usando Spinoz
        """
        velocity = velocity_kmh / 3.6
        angle_rad = np.radians(angle_deg)
        
        # Relación de expansión del difusor
        expansion_ratio = 1 + np.tan(angle_rad) * 2  # Aproximación
        
        # Spinoz del difusor
        s_diffuser = self.analyzer.calculate_spinoz(1/expansion_ratio)
        
        # Eficiencia del difusor
        if abs(s_diffuser) < 1:
            efficiency = (1 - s_diffuser) * np.cos(angle_rad)
        else:
            efficiency = 0.85  # Máximo práctico
        
        # Recuperación de presión
        pressure_recovery = efficiency * 0.5 * self.f1_data['air_density'] * velocity**2
        
        return {
            'expansion_ratio': expansion_ratio,
            'spinoz_diffuser': s_diffuser,
            'efficiency': efficiency,
            'pressure_recovery': pressure_recovery,
            'downforce_contribution': pressure_recovery * self.f1_data['frontal_area']
        }
    
    def wake_turbulence_analysis(self, lead_car_speed: float, 
                                separation_distance: float) -> Dict:
        """
        Análisis de turbulencia en la estela (dirty air)
        """
        velocity = lead_car_speed / 3.6
        
        # Longitud característica de la estela
        wake_length = self.f1_data['car_length'] * 10  # Aproximación
        
        # Decaimiento de la turbulencia con la distancia
        distance_ratio = separation_distance / self.f1_data['car_length']
        
        # Spinoz de la estela
        s_wake = self.analyzer.calculate_spinoz(1/distance_ratio)
        
        # Intensidad turbulenta en la estela
        if abs(s_wake) < 1:
            turbulence_intensity = 0.3 / (1 - s_wake)
        else:
            turbulence_intensity = 0.05  # Mínimo en aire limpio
        
        # Pérdida de downforce
        downforce_loss = turbulence_intensity * 100  # Porcentaje
        
        # Aumento de drag
        drag_increase = turbulence_intensity * 50  # Porcentaje
        
        return {
            'separation_distance': separation_distance,
            'distance_ratio': distance_ratio,
            'spinoz_wake': s_wake,
            'turbulence_intensity': turbulence_intensity,
            'downforce_loss_percent': downforce_loss,
            'drag_increase_percent': drag_increase,
            'following_difficulty': 'High' if turbulence_intensity > 0.2 else 'Moderate'
        }
    
    def track_specific_optimization(self, track_name: str) -> Dict:
        """
        Optimización específica por circuito usando Spinoz
        """
        if track_name not in self.f1_data['track_conditions']:
            raise ValueError(f"Track {track_name} not in database")
        
        track = self.f1_data['track_conditions'][track_name]
        
        # Eficiencia L/D requerida
        efficiency_required = track['efficiency']
        
        # Spinoz de eficiencia
        s_efficiency = self.analyzer.calculate_spinoz(1 - efficiency_required)
        
        # Configuración aerodinámica óptima
        if s_efficiency < 0.7:  # Alta downforce (Monaco)
            wing_angle_front = 35  # grados
            wing_angle_rear = 30
            ride_height = 30  # mm
        elif s_efficiency > 0.9:  # Baja drag (Monza)
            wing_angle_front = 10
            wing_angle_rear = 5
            ride_height = 50
        else:  # Balanceado
            wing_angle_front = 22
            wing_angle_rear = 18
            ride_height = 40
        
        # Análisis de velocidad promedio
        avg_speed = track['avg_speed']
        Re_avg = avg_speed/3.6 * self.f1_data['car_length'] / (1.5e-5)
        scales = self.analyzer.kolmogorov_scales(Re_avg, self.f1_data['car_length'], 1.5e-5)
        
        return {
            'track': track_name,
            'efficiency_required': efficiency_required,
            'spinoz_efficiency': s_efficiency,
            'wing_angle_front': wing_angle_front,
            'wing_angle_rear': wing_angle_rear,
            'ride_height': ride_height,
            'avg_reynolds': Re_avg,
            'spinoz_flow': scales['spinoz'],
            'optimization': 'Downforce' if s_efficiency < 0.7 else 'Speed'
        }

# ============================================================================
# FUNCIONES DE ANÁLISIS Y VISUALIZACIÓN
# ============================================================================

def analyze_reynolds_sweep(Re_range: np.ndarray) -> pd.DataFrame:
    """
    Análisis paramétrico barriendo números de Reynolds
    """
    analyzer = SpinozTurbulenceAnalyzer()
    results = []
    
    for Re in Re_range:
        # Escalas de Kolmogorov (usando valores típicos)
        L = 0.1  # m
        nu = 1e-6  # m²/s (agua)
        
        scales = analyzer.kolmogorov_scales(Re, L, nu)
        transition = analyzer.transition_criterion(Re)
        
        # Viscosidad turbulenta
        nu_t = analyzer.turbulent_viscosity_spinoz(nu, scales['spinoz'])
        
        results.append({
            'Reynolds': Re,
            'Spinoz': scales['spinoz'],
            'eta/L': scales['eta_over_L'],
            'Regime': transition['regime'],
            'nu_t/nu': nu_t/nu,
            'Cascade_Terms': int(1/(1-scales['spinoz'])) if scales['spinoz'] < 1 else 1000
        })
    
    return pd.DataFrame(results)

def analyze_f1_conditions() -> Dict:
    """
    Análisis completo de condiciones de F1
    """
    f1_analyzer = F1AerodynamicsSpinoz()
    results = {}
    
    # Análisis a diferentes velocidades
    speeds = [100, 200, 300, 350]  # km/h
    
    for speed in speeds:
        # Ala delantera
        front_wing = f1_analyzer.wing_reynolds_analysis(speed, 0.5)
        
        # Efecto suelo
        ground = f1_analyzer.ground_effect_analysis(speed, 0.05)
        
        # Difusor
        diffuser = f1_analyzer.diffuser_performance(speed, 10)
        
        # Estela (a 10m de distancia)
        wake = f1_analyzer.wake_turbulence_analysis(speed, 10)
        
        results[f'{speed}_kmh'] = {
            'front_wing_Re': front_wing['reynolds'],
            'front_wing_spinoz': front_wing['spinoz'],
            'ground_effect_amplification': ground['downforce_amplification'],
            'diffuser_efficiency': diffuser['efficiency'],
            'wake_downforce_loss': wake['downforce_loss_percent']
        }
    
    # Análisis por circuito
    tracks = ['monaco', 'monza', 'silverstone']
    
    for track in tracks:
        track_setup = f1_analyzer.track_specific_optimization(track)
        results[f'track_{track}'] = track_setup
    
    return results

def plot_spinoz_cascade(s_values: np.ndarray, n_terms: int = 10):
    """
    Visualiza la cascada de energía para diferentes valores de Spinoz
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    analyzer = SpinozTurbulenceAnalyzer()
    
    for idx, s in enumerate(s_values[:4]):
        ax = axes[idx//2, idx%2]
        
        if abs(s) < 1:
            terms, partial_sums, exact = analyzer.taylor_series_expansion(s, n_terms)
            title = f'Serie de Taylor (s={s:.3f})'
            series_type = 'Taylor'
        else:
            try:
                terms, partial_sums, exact = analyzer.laurent_series_expansion(s, n_terms)
                title = f'Serie de Laurent (s={s:.3f})'
                series_type = 'Laurent'
            except:
                continue
        
        x = np.arange(n_terms)
        
        # Términos individuales
        ax.bar(x, terms, alpha=0.6, label='Términos')
        
        # Suma parcial
        ax.plot(x, partial_sums, 'r-o', label='Suma acumulada')
        
        # Valor exacto
        ax.axhline(y=exact, color='g', linestyle='--', label=f'Exacto: {exact:.3f}')
        
        ax.set_xlabel('Término n')
        ax.set_ylabel('Valor')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Cascada de Energía - Representación Spinoz', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def generate_comprehensive_report():
    """
    Genera un reporte completo del análisis
    """
    print("=" * 80)
    print("ANÁLISIS COMPUTACIONAL SPINOZ-TURBULENCIA")
    print("=" * 80)
    
    # Análisis de Reynolds
    print("\n1. ANÁLISIS PARAMÉTRICO DE REYNOLDS")
    print("-" * 40)
    Re_range = np.logspace(3, 6, 20)
    df_reynolds = analyze_reynolds_sweep(Re_range)
    
    # Mostrar transiciones clave
    transitions = df_reynolds[df_reynolds['Regime'].ne(df_reynolds['Regime'].shift())]
    print("\nTransiciones de régimen:")
    print(transitions[['Reynolds', 'Spinoz', 'Regime']].to_string())
    
    # Estadísticas por régimen
    print("\nEstadísticas por régimen:")
    regime_stats = df_reynolds.groupby('Regime')['Spinoz'].agg(['mean', 'min', 'max'])
    print(regime_stats)
    
    # Análisis F1
    print("\n2. ANÁLISIS DE AERODINÁMICA F1")
    print("-" * 40)
    f1_results = analyze_f1_conditions()
    
    # Resultados por velocidad
    print("\nEfectos aerodinámicos por velocidad:")
    speed_df = pd.DataFrame({
        speed: data for speed, data in f1_results.items() 
        if 'kmh' in speed
    }).T
    print(speed_df.to_string())
    
    # Configuración por circuito
    print("\nConfiguración óptima por circuito:")
    for key, value in f1_results.items():
        if 'track' in key:
            print(f"\n{value['track'].upper()}:")
            print(f"  - Eficiencia requerida: {value['efficiency_required']:.2f}")
            print(f"  - Spinoz eficiencia: {value['spinoz_efficiency']:.3f}")
            print(f"  - Ángulos de ala: {value['wing_angle_front']}°/{value['wing_angle_rear']}°")
            print(f"  - Altura: {value['ride_height']} mm")
            print(f"  - Optimización: {value['optimization']}")
    
    # Análisis de intermitencia
    print("\n3. CORRECCIÓN POR INTERMITENCIA")
    print("-" * 40)
    analyzer = SpinozTurbulenceAnalyzer()
    
    s_base = 0.999  # Turbulencia desarrollada
    s_corrected, factor = analyzer.intermittency_correction(s_base, 0.20)
    
    print(f"Spinoz base: {s_base:.4f}")
    print(f"Factor de intermitencia: {factor:.2f}")
    print(f"Spinoz corregido: {s_corrected:.4f}")
    
    # Modelo de backscatter
    print("\n4. ANÁLISIS DE BACKSCATTER")
    print("-" * 40)
    epsilon = 1.0  # W/kg (normalizado)
    
    for s in [0.5, 0.8, 0.99, 0.999]:
        backscatter = analyzer.backscatter_model(epsilon, s)
        print(f"\ns = {s:.3f}:")
        print(f"  - Transferencia directa: {backscatter['forward_transfer']:.3f}")
        print(f"  - Backscatter: {backscatter['backward_transfer']:.3f}")
        print(f"  - Fracción backscatter: {backscatter['backscatter_fraction']:.1%}")
    
    # Comparación con datos experimentales
    print("\n5. VALIDACIÓN CON DATOS EXPERIMENTALES")
    print("-" * 40)
    
    # Kolmogorov -5/3
    s_kolmogorov = analyzer.calculate_spinoz(3/5)
    print(f"Exponente Kolmogorov -5/3 → Spinoz = {s_kolmogorov:.3f}")
    
    # Reynolds crítico
    Re_crit = 2300
    scales_crit = analyzer.kolmogorov_scales(Re_crit, 0.1, 1e-6)
    print(f"Reynolds crítico {Re_crit} → Spinoz = {scales_crit['spinoz']:.4f}")
    
    # Bacterial turbulence
    s_bacterial = 1 - 0.20
    print(f"Bacterial turbulence (μ=0.20) → Spinoz = {s_bacterial:.3f}")
    
    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETADO")
    print("=" * 80)

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Generar reporte completo
    generate_comprehensive_report()
    
    # Crear visualizaciones
    print("\nGenerando visualizaciones...")
    
    # Visualización de cascadas
    s_values = np.array([0.4, 0.8, 0.99, 1.5])
    fig1 = plot_spinoz_cascade(s_values)
    plt.savefig('spinoz_cascade_analysis.png', dpi=150, bbox_inches='tight')
    print("Guardado: spinoz_cascade_analysis.png")
    
    # Análisis de barrido de Reynolds
    Re_range = np.logspace(2, 7, 100)
    df = analyze_reynolds_sweep(Re_range)
    
    # Gráfico de Spinoz vs Reynolds
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Spinoz vs Reynolds
    ax1.semilogx(df['Reynolds'], df['Spinoz'], 'b-', linewidth=2)
    ax1.axhline(y=0.996, color='r', linestyle='--', label='Transición crítica')
    ax1.set_xlabel('Reynolds Number')
    ax1.set_ylabel('Spinoz Parameter')
    ax1.set_title('Parámetro Spinoz vs Reynolds')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Viscosidad turbulenta
    ax2.loglog(df['Reynolds'], df['nu_t/nu'], 'g-', linewidth=2)
    ax2.set_xlabel('Reynolds Number')
    ax2.set_ylabel('νt/ν (Viscosidad turbulenta relativa)')
    ax2.set_title('Amplificación de Viscosidad Turbulenta')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spinoz_reynolds_analysis.png', dpi=150, bbox_inches='tight')
    print("Guardado: spinoz_reynolds_analysis.png")
    
    # Análisis F1 detallado
    print("\nGenerando análisis F1 detallado...")
    f1_analyzer = F1AerodynamicsSpinoz()
    
    # Comparación de circuitos
    fig3, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    tracks = ['monaco', 'monza', 'silverstone']
    speeds = np.linspace(50, 350, 50)
    
    for idx, track in enumerate(tracks):
        track_data = f1_analyzer.track_specific_optimization(track)
        
        # Análisis de downforce vs velocidad
        ax1 = axes[0, idx]
        downforce = []
        drag = []
        
        for speed in speeds:
            ground = f1_analyzer.ground_effect_analysis(speed, track_data['ride_height']/1000)
            downforce.append(ground['effective_downforce'])
            drag.append(f1_analyzer.f1_data['drag_coefficient'] * (1 + ground['spinoz_ground']))
        
        ax1.plot(speeds, downforce, 'b-', label='Downforce', linewidth=2)
        ax1.plot(speeds, drag, 'r-', label='Drag', linewidth=2)
        ax1.set_xlabel('Velocidad (km/h)')
        ax1.set_ylabel('Coeficiente')
        ax1.set_title(f'{track.capitalize()}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Análisis de estela
        ax2 = axes[1, idx]
        distances = np.linspace(1, 50, 50)
        downforce_loss = []
        
        for dist in distances:
            wake = f1_analyzer.wake_turbulence_analysis(
                track_data['avg_reynolds']*1.5e-5/f1_analyzer.f1_data['car_length']*3.6, 
                dist
            )
            downforce_loss.append(wake['downforce_loss_percent'])
        
        ax2.plot(distances, downforce_loss, 'g-', linewidth=2)
        ax2.set_xlabel('Distancia (m)')
        ax2.set_ylabel('Pérdida Downforce (%)')
        ax2.set_title(f'Efecto Estela - {track.capitalize()}')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Límite crítico')
        ax2.legend()
    
    plt.suptitle('Análisis Aerodinámico F1 por Circuito', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('f1_circuit_analysis.png', dpi=150, bbox_inches='tight')
    print("Guardado: f1_circuit_analysis.png")
    
    # Mapa de calor Spinoz para diferentes condiciones
    print("\nGenerando mapa de calor Spinoz...")
    
    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Crear instancia del analizador
    analyzer = SpinozTurbulenceAnalyzer()
    
    # Mapa de calor: Reynolds vs Velocidad
    Re_grid = np.logspace(3, 7, 50)
    vel_grid = np.linspace(10, 400, 50)
    
    spinoz_map = np.zeros((len(vel_grid), len(Re_grid)))
    
    for i, vel in enumerate(vel_grid):
        for j, Re in enumerate(Re_grid):
            scales = analyzer.kolmogorov_scales(Re, 1.0, 1.5e-5)
            spinoz_map[i, j] = scales['spinoz']
    
    im1 = ax1.contourf(Re_grid, vel_grid, spinoz_map, levels=20, cmap='RdYlBu_r')
    ax1.set_xscale('log')
    ax1.set_xlabel('Reynolds Number')
    ax1.set_ylabel('Velocidad (km/h)')
    ax1.set_title('Mapa de Parámetro Spinoz')
    plt.colorbar(im1, ax=ax1, label='Spinoz s')
    
    # Contornos críticos
    ax1.contour(Re_grid, vel_grid, spinoz_map, levels=[0.996], colors='white', linewidths=2)
    
    # Mapa de calor: Eficiencia aerodinámica
    clearance_grid = np.linspace(0.02, 0.15, 50)
    angle_grid = np.linspace(5, 20, 50)
    
    efficiency_map = np.zeros((len(angle_grid), len(clearance_grid)))
    
    for i, angle in enumerate(angle_grid):
        for j, clearance in enumerate(clearance_grid):
            ground = f1_analyzer.ground_effect_analysis(250, clearance)
            diffuser = f1_analyzer.diffuser_performance(250, angle)
            efficiency_map[i, j] = ground['downforce_amplification'] * diffuser['efficiency']
    
    im2 = ax2.contourf(clearance_grid*1000, angle_grid, efficiency_map, levels=20, cmap='viridis')
    ax2.set_xlabel('Altura (mm)')
    ax2.set_ylabel('Ángulo Difusor (°)')
    ax2.set_title('Eficiencia Aerodinámica Combinada')
    plt.colorbar(im2, ax=ax2, label='Eficiencia')
    
    plt.tight_layout()
    plt.savefig('spinoz_heat_maps.png', dpi=150, bbox_inches='tight')
    print("Guardado: spinoz_heat_maps.png")
    
    print("\n✓ Análisis completo finalizado")