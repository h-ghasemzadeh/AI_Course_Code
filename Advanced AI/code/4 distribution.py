import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (norm, expon, lognorm, gamma, beta, chi2, 
                         weibull_min, gumbel_r, kstest, anderson)
from sklearn.metrics import mean_squared_error

class DistributionFitter:
    def __init__(self, data):
        self.data = data
        self.results = {}
        
    def fit_distributions(self):
        """Fit data to various distributions and store results"""
        distributions = [
            ('Normal', norm, self._fit_normal),
            ('Exponential', expon, self._fit_exponential),
            ('Log-Normal', lognorm, self._fit_lognormal),
            ('Gamma', gamma, self._fit_gamma),
            ('Beta', beta, self._fit_beta),
            ('Weibull', weibull_min, self._fit_weibull),
            ('Gumbel', gumbel_r, self._fit_gumbel)
        ]
        
        for name, dist, fit_func in distributions:
            try:
                params, ks_stat, ks_p, ad_stat, ad_crit, mse = fit_func()
                self.results[name] = {
                    'params': params,
                    'ks_stat': ks_stat,
                    'ks_p': ks_p,
                    'ad_stat': ad_stat,
                    'ad_crit': ad_crit,
                    'mse': mse
                }
            except Exception as e:
                print(f"Error fitting {name}: {str(e)}")
                
        return self.results
    
    def _fit_normal(self):
        """Fit normal distribution"""
        mean, std = norm.fit(self.data)
        params = (mean, std)
        ks_stat, ks_p = kstest(self.data, 'norm', args=params)
        ad_result = anderson(self.data, 'norm')
        mse = self._calculate_mse(norm, params)
        return params, ks_stat, ks_p, ad_result.statistic, ad_result.critical_values[2], mse
    
    def _fit_exponential(self):
        """Fit exponential distribution"""
        loc, scale = expon.fit(self.data)
        params = (loc, scale)
        ks_stat, ks_p = kstest(self.data, 'expon', args=params)
        ad_result = anderson(self.data, 'expon')
        mse = self._calculate_mse(expon, params)
        return params, ks_stat, ks_p, ad_result.statistic, ad_result.critical_values[2], mse
    
    def _fit_lognormal(self):
        """Fit log-normal distribution"""
        shape, loc, scale = lognorm.fit(self.data)
        params = (shape, loc, scale)
        ks_stat, ks_p = kstest(self.data, 'lognorm', args=params)
        ad_result = anderson(self.data, 'lognorm')
        mse = self._calculate_mse(lognorm, params)
        return params, ks_stat, ks_p, ad_result.statistic, ad_result.critical_values[2], mse
    
    def _fit_gamma(self):
        """Fit gamma distribution"""
        a, loc, scale = gamma.fit(self.data)
        params = (a, loc, scale)
        ks_stat, ks_p = kstest(self.data, 'gamma', args=params)
        ad_result = anderson(self.data, 'gamma')
        mse = self._calculate_mse(gamma, params)
        return params, ks_stat, ks_p, ad_result.statistic, ad_result.critical_values[2], mse
    
    def _fit_beta(self):
        """Fit beta distribution"""
        a, b, loc, scale = beta.fit(self.data)
        params = (a, b, loc, scale)
        ks_stat, ks_p = kstest(self.data, 'beta', args=params)
        # Anderson-Darling test not available for beta distribution
        mse = self._calculate_mse(beta, params)
        return params, ks_stat, ks_p, np.nan, np.nan, mse
    
    def _fit_weibull(self):
        """Fit Weibull distribution"""
        c, loc, scale = weibull_min.fit(self.data)
        params = (c, loc, scale)
        ks_stat, ks_p = kstest(self.data, 'weibull_min', args=params)
        ad_result = anderson(self.data, 'weibull_min')
        mse = self._calculate_mse(weibull_min, params)
        return params, ks_stat, ks_p, ad_result.statistic, ad_result.critical_values[2], mse
    
    def _fit_gumbel(self):
        """Fit Gumbel distribution"""
        loc, scale = gumbel_r.fit(self.data)
        params = (loc, scale)
        ks_stat, ks_p = kstest(self.data, 'gumbel_r', args=params)
        # Anderson-Darling test not available for Gumbel distribution
        mse = self._calculate_mse(gumbel_r, params)
        return params, ks_stat, ks_p, np.nan, np.nan, mse
    
    def _calculate_mse(self, dist, params):
        """Calculate mean squared error"""
        hist, bin_edges = np.histogram(self.data, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        pdf_values = dist.pdf(bin_centers, *params)
        return mean_squared_error(hist, pdf_values)
    
    def get_best_distribution(self):
        """Select best distribution based on different criteria"""
        if not self.results:
            self.fit_distributions()
            
        # Selection criteria
        best_ks = min(self.results.items(), key=lambda x: x[1]['ks_stat'])
        best_ad = min((item for item in self.results.items() if not np.isnan(item[1]['ad_stat'])), 
                     key=lambda x: x[1]['ad_stat'], default=(None, None))
        best_mse = min(self.results.items(), key=lambda x: x[1]['mse'])
        
        print("\nBest Distributions:")
        print(f"By Kolmogorov-Smirnov: {best_ks[0]} (stat={best_ks[1]['ks_stat']:.4f})")
        if best_ad[0]:
            print(f"By Anderson-Darling: {best_ad[0]} (stat={best_ad[1]['ad_stat']:.4f})")
        print(f"By MSE: {best_mse[0]} (MSE={best_mse[1]['mse']:.4f})")
        
        return {
            'Kolmogorov-Smirnov': best_ks,
            'Anderson-Darling': best_ad,
            'MSE': best_mse
        }
    
    def plot_results(self):
        """Plot data histogram and best fitted distributions"""
        plt.figure(figsize=(12, 8))
        hist, bins, _ = plt.hist(self.data, bins='auto', density=True, alpha=0.6, color='b')
        
        x = np.linspace(min(self.data), max(self.data), 1000)
        
        # Plot top 3 distributions
        best_dists = self.get_best_distribution()
        for i, (name, result) in enumerate(best_dists.items()):
            if result[0] is None:
                continue
                
            dist_name = result[0]
            params = result[1]['params']
            
            if dist_name == 'Normal':
                pdf = norm.pdf(x, *params)
            elif dist_name == 'Exponential':
                pdf = expon.pdf(x, *params)
            elif dist_name == 'Log-Normal':
                pdf = lognorm.pdf(x, *params)
            elif dist_name == 'Gamma':
                pdf = gamma.pdf(x, *params)
            elif dist_name == 'Beta':
                pdf = beta.pdf(x, *params)
            elif dist_name == 'Weibull':
                pdf = weibull_min.pdf(x, *params)
            elif dist_name == 'Gumbel':
                pdf = gumbel_r.pdf(x, *params)
            
            plt.plot(x, pdf, label=f"{dist_name} ({name})", linewidth=2)
        
        plt.title('Data Distribution and Best Fitted Distributions')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

# Generate sample data
np.random.seed(42)
data = np.concatenate([np.random.normal(5, 2, 2000), 
                       np.random.exponential(1, 1000)])

# Run analysis
fitter = DistributionFitter(data)
fitter.fit_distributions()
best_dists = fitter.get_best_distribution()
fitter.plot_results()

# Display detailed results for all distributions
print("\nDetailed Results for All Distributions:")
for name, result in fitter.results.items():
    print(f"\n{name}:")
    print(f"Parameters: {result['params']}")
    print(f"K-S Stat: {result['ks_stat']:.4f}, p-value: {result['ks_p']:.4f}")
    print(f"A-D Stat: {result['ad_stat']:.4f}, Critical: {result['ad_crit']:.4f}" 
          if not np.isnan(result['ad_stat']) else "A-D Test: Not available")
    print(f"MSE: {result['mse']:.4f}")