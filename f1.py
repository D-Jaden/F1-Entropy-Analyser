import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ENABLE CACHE
fastf1.Cache.enable_cache('cache')

class F1PerformanceEntropyAnalyzer:
    def __init__(self, year=2021, race='Abu Dhabi'): #SET IT UP FOR YOUR DESIRED RACE
        self.year = year
        self.race = race
        self.session = None
        self.entropy_scores = {}
        
    def load_race_data(self):
        print(f"Loading {self.race} {self.year} race data...")
        self.session = fastf1.get_session(self.year, self.race, 'R')
        self.session.load()
        print("Data loaded successfully!")
        
    def get_driver_info(self, driver_code):
        try:
            driver_info = self.session.get_driver(driver_code)
            driver_name = f"{driver_info['FirstName']} {driver_info['LastName']}"
            team_name = driver_info['TeamName']
            return driver_name, team_name
        except:
            return driver_code, "Unknown"
    
    def calculate_sector_entropy(self, driver_code, normalize=True):
        """Calculate entropy for each sector based on lap time variations 
        Higher entropy = more inconsistent performance"""
        driver_laps = self.session.laps.pick_driver(driver_code)
        
        # Filter out outliers and invalid laps
        driver_laps = driver_laps[
            (driver_laps['Sector1Time'].notna()) &
            (driver_laps['Sector2Time'].notna()) &
            (driver_laps['Sector3Time'].notna()) &
            (driver_laps['LapTime'].notna()) &
            (driver_laps['IsAccurate'] == True)  # Only accurate laps
        ]
        
        if len(driver_laps) < 5:
            return None
            
        # Convert sector times to seconds
        sector1_times = driver_laps['Sector1Time'].dt.total_seconds()
        sector2_times = driver_laps['Sector2Time'].dt.total_seconds()
        sector3_times = driver_laps['Sector3Time'].dt.total_seconds()
        lap_times = driver_laps['LapTime'].dt.total_seconds()
        
        # Calculate normalized entropy for each sector
        def calculate_entropy(times):
            if len(times) < 3:
                return 0
            # Remove extreme outliers using IQR method
            q75, q25 = np.percentile(times, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            times = times[(times >= lower_bound) & (times <= upper_bound)]
            
            if len(times) < 3:
                return 0
                
            # Calculate coefficient of variation (std/mean) as entropy measure
            return np.std(times) / np.mean(times) if np.mean(times) > 0 else 0
        
        sector_entropies = {
            'sector1': calculate_entropy(sector1_times),
            'sector2': calculate_entropy(sector2_times),
            'sector3': calculate_entropy(sector3_times),
            'overall': calculate_entropy(lap_times),
            'lap_count': len(driver_laps),
            'avg_laptime': np.mean(lap_times)
        }
        
        return sector_entropies
    
    def calculate_pace_analysis(self, driver_code):
        """
        Analyze pace consistency and degradation throughout the race
        """
        driver_laps = self.session.laps.pick_driver(driver_code)
        valid_laps = driver_laps[
            (driver_laps['LapTime'].notna()) &
            (driver_laps['IsAccurate'] == True)
        ]
        
        if len(valid_laps) < 5:
            return None
            
        lap_times = valid_laps['LapTime'].dt.total_seconds()
        lap_numbers = valid_laps['LapNumber'].values
        
        # Calculate pace degradation (linear trend)
        if len(lap_times) > 5:
            slope, intercept, r_value, p_value, std_err = stats.linregress(lap_numbers, lap_times)
            degradation_rate = slope  # seconds per lap
        else:
            degradation_rate = 0
            
        # Calculate stint consistency (group by compound if available)
        stint_consistency = np.std(lap_times) / np.mean(lap_times)
        
        # Find best and worst sectors
        fastest_lap_idx = lap_times.idxmin()
        fastest_lap = valid_laps.loc[fastest_lap_idx]
        
        return {
            'degradation_rate': degradation_rate,
            'stint_consistency': stint_consistency,
            'fastest_laptime': fastest_lap['LapTime'].total_seconds(),
            'fastest_lap_number': fastest_lap['LapNumber'],
            'pace_variance': np.var(lap_times)
        }
    
    def analyze_race_position_consistency(self, driver_code):
        """
        Analyze how consistent driver's race position was
        """
        driver_laps = self.session.laps.pick_driver(driver_code)
        valid_laps = driver_laps[driver_laps['Position'].notna()]
        
        if len(valid_laps) < 5:
            return None
            
        positions = valid_laps['Position'].values
        
        # Calculate position variance and range
        position_variance = np.var(positions)
        position_range = np.max(positions) - np.min(positions)
        position_trend = np.mean(np.diff(positions))  # Positive = losing positions
        
        return {
            'position_variance': position_variance,
            'position_range': position_range,
            'position_trend': position_trend,
            'final_position': positions[-1] if len(positions) > 0 else None,
            'starting_position': positions[0] if len(positions) > 0 else None
        }
    
    def create_performance_fingerprint(self, driver_code):
        """
        Create a comprehensive performance fingerprint using available data
        """
        # Get basic driver info
        driver_name, team_name = self.get_driver_info(driver_code)
        
        # Calculate sector entropy
        sector_entropy = self.calculate_sector_entropy(driver_code)
        if not sector_entropy:
            print(f"No valid sector data for {driver_code}")
            return None
            
        # Calculate pace analysis
        pace_analysis = self.calculate_pace_analysis(driver_code)
        if not pace_analysis:
            print(f"No valid pace data for {driver_code}")
            return None
            
        # Calculate position consistency
        position_analysis = self.analyze_race_position_consistency(driver_code)
        
        fingerprint = {
            'driver_code': driver_code,
            'driver_name': driver_name,
            'team': team_name,
            'consistency_score': 1 / (1 + sector_entropy['overall']),
            'sector1_chaos': sector_entropy['sector1'],
            'sector2_chaos': sector_entropy['sector2'], 
            'sector3_chaos': sector_entropy['sector3'],
            'overall_chaos': sector_entropy['overall'],
            'lap_count': sector_entropy['lap_count'],
            'avg_laptime': sector_entropy['avg_laptime'],
            'degradation_rate': pace_analysis['degradation_rate'],
            'stint_consistency': pace_analysis['stint_consistency'],
            'fastest_laptime': pace_analysis['fastest_laptime'],
            'pace_variance': pace_analysis['pace_variance']
        }
        
        # Add position analysis if available
        if position_analysis:
            fingerprint.update({
                'position_variance': position_analysis['position_variance'],
                'position_range': position_analysis['position_range'],
                'final_position': position_analysis['final_position'],
                'starting_position': position_analysis['starting_position']
            })
        else:
            fingerprint.update({
                'position_variance': 0,
                'position_range': 0,
                'final_position': None,
                'starting_position': None
            })
            
        return fingerprint
    
    def analyze_all_drivers(self):
        """Analyze all drivers in the race"""
        drivers = self.session.drivers
        fingerprints = []
        
        print("Analyzing drivers:")
        for driver in drivers:
            print(f"  Processing {driver}...")
            fingerprint = self.create_performance_fingerprint(driver)
            if fingerprint:
                fingerprints.append(fingerprint)
                
        return pd.DataFrame(fingerprints)
    
    def visualize_driver_fingerprints(self, df):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'F1 Driver Performance Analysis - {self.race} {self.year}', fontsize=16)
        
        # 1. Consistency vs Overall Chaos
        scatter = axes[0,0].scatter(df['consistency_score'], df['overall_chaos'], 
                                  s=100, alpha=0.7, c=df.index, cmap='viridis')
        axes[0,0].set_xlabel('Consistency Score (Higher = More Consistent)')
        axes[0,0].set_ylabel('Overall Chaos (Higher = More Variable)')
        axes[0,0].set_title('Consistency vs Variability')
        
        # Add driver labels
        for i, row in df.iterrows():
            axes[0,0].annotate(row['driver_code'], 
                              (row['consistency_score'], row['overall_chaos']),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 2. Sector chaos comparison
        sector_data = df[['driver_code', 'sector1_chaos', 'sector2_chaos', 'sector3_chaos']].set_index('driver_code')
        sector_data.plot(kind='bar', ax=axes[0,1], width=0.8, rot=45)
        axes[0,1].set_title('Sector-by-Sector Variability')
        axes[0,1].set_xlabel('Driver')
        axes[0,1].set_ylabel('Chaos Level')
        axes[0,1].legend(['Sector 1', 'Sector 2', 'Sector 3'])
        
        # 3. Pace degradation analysis
        valid_degradation = df[df['degradation_rate'].notna()]
        if len(valid_degradation) > 0:
            bars = axes[0,2].bar(range(len(valid_degradation)), valid_degradation['degradation_rate'])
            axes[0,2].set_title('Pace Degradation Rate')
            axes[0,2].set_xlabel('Driver Index')
            axes[0,2].set_ylabel('Degradation (sec/lap)')
            axes[0,2].set_xticks(range(len(valid_degradation)))
            axes[0,2].set_xticklabels(valid_degradation['driver_code'], rotation=45)
            
            # Color bars: green for improvement, red for degradation
            for i, bar in enumerate(bars):
                if valid_degradation.iloc[i]['degradation_rate'] > 0:
                    bar.set_color('red')
                else:
                    bar.set_color('green')
        
        # 4. Average lap time vs consistency
        axes[1,0].scatter(df['avg_laptime'], df['consistency_score'], 
                         s=100, alpha=0.7, c=df.index, cmap='plasma')
        axes[1,0].set_xlabel('Average Lap Time (seconds)')
        axes[1,0].set_ylabel('Consistency Score')
        axes[1,0].set_title('Speed vs Consistency Trade-off')
        
        for i, row in df.iterrows():
            axes[1,0].annotate(row['driver_code'], 
                              (row['avg_laptime'], row['consistency_score']),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 5. Position analysis (if available)
        position_data = df[df['final_position'].notna()]
        if len(position_data) > 0:
            axes[1,1].scatter(position_data['starting_position'], position_data['final_position'],
                             s=100, alpha=0.7)
            axes[1,1].plot([1, 20], [1, 20], 'r--', alpha=0.5)  # Reference line
            axes[1,1].set_xlabel('Starting Position')
            axes[1,1].set_ylabel('Final Position')
            axes[1,1].set_title('Starting vs Final Position')
            axes[1,1].invert_yaxis()
            axes[1,1].invert_xaxis()
            
            for i, row in position_data.iterrows():
                axes[1,1].annotate(row['driver_code'],
                                  (row['starting_position'], row['final_position']),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 6. Performance summary heatmap
        heatmap_cols = ['consistency_score', 'stint_consistency', 'overall_chaos']
        heatmap_data = df[heatmap_cols].copy()
        
        # Normalize for heatmap
        for col in heatmap_cols:
            if col == 'overall_chaos':  # Lower is better
                heatmap_data[col] = 1 - (heatmap_data[col] - heatmap_data[col].min()) / \
                                   (heatmap_data[col].max() - heatmap_data[col].min())
            else:  # Higher is better
                heatmap_data[col] = (heatmap_data[col] - heatmap_data[col].min()) / \
                                   (heatmap_data[col].max() - heatmap_data[col].min())
        
        im = axes[1,2].imshow(heatmap_data.T, cmap='RdYlGn', aspect='auto')
        axes[1,2].set_title('Performance Summary')
        axes[1,2].set_xlabel('Driver Index')
        axes[1,2].set_yticks(range(len(heatmap_cols)))
        axes[1,2].set_yticklabels(['Consistency', 'Stint Quality', 'Low Chaos'])
        axes[1,2].set_xticks(range(len(df)))
        axes[1,2].set_xticklabels(df['driver_code'], rotation=45)
        
        plt.colorbar(im, ax=axes[1,2], label='Performance Score')
        plt.tight_layout()
        return fig
    
    def find_performance_insights(self, df):
        """Generate interesting insights from the data"""
        insights = []
        
        if len(df) == 0:
            return ["No data available for analysis"]
        
        # Most consistent driver
        most_consistent = df.loc[df['consistency_score'].idxmax()]
        insights.append(f"Most Consistent: {most_consistent['driver_name']} ({most_consistent['team']}) - Score: {most_consistent['consistency_score']:.3f}")
        
        # Most chaotic driver
        most_chaotic = df.loc[df['overall_chaos'].idxmax()]
        insights.append(f"Most Variable: {most_chaotic['driver_name']} ({most_chaotic['team']}) - Chaos: {most_chaotic['overall_chaos']:.3f}")
        
        # Best pace degradation (if available)
        valid_deg = df[df['degradation_rate'].notna()]
        if len(valid_deg) > 0:
            best_deg = valid_deg.loc[valid_deg['degradation_rate'].idxmin()]
            if best_deg['degradation_rate'] < 0:
                insights.append(f"Best Pace Improvement: {best_deg['driver_name']} ({best_deg['degradation_rate']:.3f} sec/lap faster over race)")
            else:
                insights.append(f"Best Pace Maintenance: {best_deg['driver_name']} ({best_deg['degradation_rate']:.3f} sec/lap degradation)")
        
        # Fastest driver
        fastest = df.loc[df['fastest_laptime'].idxmin()]
        insights.append(f"Fastest Lap: {fastest['driver_name']} - {fastest['fastest_laptime']:.3f} seconds")
        
        # Sector specialists
        sector_specialists = []
        for sector_num in [1, 2, 3]:
            col = f'sector{sector_num}_chaos'
            specialist = df.loc[df[col].idxmin()]
            sector_specialists.append(f"Sector {sector_num}: {specialist['driver_name']} ({specialist[col]:.3f})")
        
        insights.append("Sector Specialists (lowest variability): " + ", ".join(sector_specialists))
        
        return insights

def run_entropy_analysis(year=2021, race='Abu Dhabi'):
    """Run the complete entropy analysis"""
    try:
        # Initialize analyzer
        analyzer = F1PerformanceEntropyAnalyzer(year=year, race=race)
        
        # Load data
        analyzer.load_race_data()
        
        # Analyze all drivers
        results_df = analyzer.analyze_all_drivers()
        
        if results_df.empty:
            print("No valid data found for analysis")
            return None, None
        
        # Display results
        print(f"\n=== DRIVER PERFORMANCE ANALYSIS - {race} {year} ===")
        print(f"Analyzed {len(results_df)} drivers")
        print("\nTop performers by consistency:")
        top_consistent = results_df.nlargest(5, 'consistency_score')[['driver_name', 'team', 'consistency_score', 'overall_chaos']]
        print(top_consistent.to_string(index=False))
        
        # Generate insights
        print("\n=== PERFORMANCE INSIGHTS ===")
        insights = analyzer.find_performance_insights(results_df)
        for insight in insights:
            print(f"• {insight}")
        
        # Create visualizations
        try:
            fig = analyzer.visualize_driver_fingerprints(results_df)
            plt.show()
        except Exception as e:
            print(f"Visualization error: {e}")
        
        return analyzer, results_df
        
    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Run the analysis
if __name__ == "__main__":
    analyzer, results = run_entropy_analysis()
    if results is not None:
        print("\nFull results:")
        print(results[['driver_name', 'team', 'consistency_score', 'overall_chaos', 'fastest_laptime']].round(3))
